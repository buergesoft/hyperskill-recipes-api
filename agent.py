from llama_index.llms.openai import OpenAIResponses
from github import Github, Auth
from typing import Any
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    FunctionAgent, AgentWorkflow,
    AgentOutput, ToolCall, ToolCallResult,
)
from llama_index.core.workflow import Context
from llama_index.core.prompts import RichPromptTemplate
import dotenv
import os
import asyncio

dotenv.load_dotenv()

git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN"))) if os.getenv("GITHUB_TOKEN") else None

repo_url = os.getenv("REPOSITORY")
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

if git is not None:
    repo = git.get_repo(full_repo_name)


def get_pr_details(pr_number: int) -> dict:
    """Useful for retrieving details about a pull request given its number.
    Returns the author, title, body, diff_url, state, and commit SHAs."""
    pull_request = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pull_request.get_commits()]
    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body or "none",
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_SHAs": commit_SHAs,
        "head_sha": commit_SHAs[-1] if commit_SHAs else None,
    }

def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Useful for retrieving details about a commit given its SHA.
    Returns the files that changed, with status, additions, deletions, changes, and patch."""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files

def get_file_contents(file_path: str) -> str:
    """Useful for fetching the contents of a file from the repository given its path."""
    return repo.get_contents(file_path).decoded_content.decode("utf-8")

async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for adding the gathered context to the current state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "Context added to state."

async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft comment to the current state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "Draft comment added to state."

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the current state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "Final review added to state."

def post_review_to_github(pr_number: int, review_comment: str) -> str:
    """Useful for posting a final review comment to a pull request on GitHub.
    Takes the PR number and the final review comment."""
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=review_comment)
    return "Review posted to GitHub."

llm = OpenAIResponses(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    reasoning_effort="medium",
)

get_pr_details_tool = FunctionTool.from_defaults(get_pr_details)
get_commit_details_tool = FunctionTool.from_defaults(get_commit_details)
get_file_contents_tool = FunctionTool.from_defaults(get_file_contents)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)


context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context about a pull request and the repository.",
    tools=[get_pr_details_tool, get_commit_details_tool,
           get_file_contents_tool, add_context_to_state],
    system_prompt=(
        """You are the context gathering agent. When gathering context, you MUST gather \n:
    - The details: author, title, body, diff_url, state, and head_sha; \n
    - Changed files; \n
    - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state],
    system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Context Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" 
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. 
 """,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment, requests rewrites if needed, and posts the final review to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.  """,
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={"gathered_contexts": "", "draft_comment": "", "final_review": ""},
)

async def main():
    pr_number = os.getenv("PR_NUMBER") or ""
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
