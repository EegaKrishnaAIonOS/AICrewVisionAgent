"""Reusable CrewAI pipeline for image + query analysis with caching."""

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List, Tuple

from crewai import Crew, LLM, Agent, Process, Task
from crewai_tools import VisionTool
from dotenv import load_dotenv
from PIL import Image

# Fix Windows console encoding issue for Unicode characters (emojis)
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()

_VISION_CACHE: dict[str, str] = {}
_VISION_CACHE_LOCK = Lock()


def _hash_file(image_path: str) -> str:
    sha = hashlib.sha256()
    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class CachedVisionTool(VisionTool):
    """VisionTool wrapper that caches descriptions per image hash."""

    def _run(self, **kwargs) -> str:
        image_path_url = kwargs.get("image_path_url")
        use_cache = False
        cache_key: str | None = None

        if image_path_url and not image_path_url.startswith("http") and os.path.exists(
            image_path_url
        ):
            cache_key = _hash_file(image_path_url)
            with _VISION_CACHE_LOCK:
                cached_value = _VISION_CACHE.get(cache_key)
            if cached_value:
                use_cache = True
                print(
                    "⚡ VisionTool cache hit: reusing previous description "
                    f"for image hash {cache_key[:8]}… (no new tokens consumed)"
                )
                return cached_value

        print(
            "ℹ️ VisionTool call will use OpenAI's gpt-4o-mini vision endpoint. "
            "Expect roughly ~750 tokens on first call for a new image."
        )
        result = super()._run(**kwargs)

        if cache_key and not use_cache and isinstance(result, str) and result.strip():
            with _VISION_CACHE_LOCK:
                _VISION_CACHE[cache_key] = result
            print(
                "✅ VisionTool cache stored result for image hash "
                f"{cache_key[:8]}…. Subsequent queries reuse it."
            )
        return result


def _validate_image_path(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    Image.open(path).close()
    return str(path)


def _ensure_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in .env file.\n"
            "VisionTool requires OpenAI API key for image analysis.\n"
            "You need BOTH API keys:\n"
            "  - GROQ_API_KEY: for agent's LLM (text generation)\n"
            "  - OPENAI_API_KEY: for VisionTool (image analysis)\n"
            "Please add OPENAI_API_KEY to your .env file."
        )


@lru_cache(maxsize=1)
def _get_llm() -> LLM:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key is None:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. Please add GROQ_API_KEY to your .env file."
        )
    os.environ["GROQ_API_KEY"] = groq_api_key
    return LLM(
        model="groq/moonshotai/kimi-k2-instruct",
        temperature=0.7,
        api_key=groq_api_key,
    )


def run_image_analysis(image_path: str | Path, user_query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Run the CrewAI pipeline for a given image and query."""
    normalized_path = _validate_image_path(image_path)
    _ensure_openai_key()
    llm = _get_llm()
    vision_tool = CachedVisionTool()

    image_analyzer_agent = Agent(
        role="Image Describer",
        goal=f"Accurately describe the image at {normalized_path}.",
        backstory="You are an expert in image analysis and scene understanding.",
        verbose=True,
        tools=[vision_tool],
        llm=llm,
    )

    # designer_agent = Agent(
    #     role="Design Improvement Specialist",
    #     goal="Suggest actionable improvements leveraging the image context.",
    #     backstory="UX specialist focused on optimizing visual experiences.",
    #     verbose=True,
    #     tools=[vision_tool],
    #     llm=llm,
    # )

    # product_manager_agent = Agent(
    #     role="Product Manager",
    #     goal="Translate insights into user stories and priorities.",
    #     backstory="Seasoned PM for restaurant-tech solutions.",
    #     verbose=True,
    #     tools=[vision_tool],
    #     llm=llm,
    # )

    # query_responder_agent = Agent(
    #     role="Query Responder",
    #     goal=f"Answer the user's question: {user_query}",
    #     backstory="Expert communicator who synthesizes insights into clear answers.",
    #     verbose=True,
    #     tools=[vision_tool],
    #     llm=llm,
    # )

    task_describe_image = Task(
        description=(
            f"Use the Vision Tool to analyze the image at '{normalized_path}'. "
            "Provide a detailed description of answer for the user query: '{user_query}'"
            f"Ensure the Vision Tool is invoked with image_path_url='{normalized_path}'."
        ),
        expected_output="Detailed description of the image for the user query: '{user_query}'.",
        agent=image_analyzer_agent,
    )

    # task_suggest_improvements = Task(
    #     description=(
    #         "Using the previous description, propose concrete improvements or insights relevant to the scene. "
    #         f"If additional clarification is needed, inspect the image via Vision Tool at '{normalized_path}'."
    #     ),
    #     expected_output="List of actionable improvements or insights.",
    #     agent=designer_agent,
    #     context=[task_describe_image],
    # )

    # task_create_user_stories = Task(
    #     description=(
    #         "Transform the gathered insights into prioritized user stories or action items. "
    #         "Mention impact and feasibility briefly."
    #     ),
    #     expected_output="Prioritized list of user stories/action items.",
    #     agent=product_manager_agent,
    #     context=[task_describe_image, task_suggest_improvements],
    # )

    # task_answer_query = Task(
    #     description=(
    #         f"Answer the user's query: '{user_query}'. Use all prior context and reference the image at "
    #         f"'{normalized_path}'. Provide a clear, concise response tailored to the query."
    #     ),
    #     expected_output="Direct answer to the user query with any supporting rationale.",
    #     agent=query_responder_agent,
    #     context=[task_describe_image, task_suggest_improvements, task_create_user_stories],
    # )

    ai_team = Crew(
        agents=[
            image_analyzer_agent,
            #designer_agent,
            #product_manager_agent,
            # query_responder_agent,
        ],
        tasks=[
            task_describe_image,
            #task_suggest_improvements,
            #task_create_user_stories,
            # task_answer_query,
        ],
        process=Process.sequential,
        verbose=True,
    )

    try:
        results = ai_team.kickoff()
    except Exception as exc:
        message = str(exc).lower()
        if "rate limit" in message or "429" in message:
            friendly_msg = (
                "VisionTool hit the OpenAI rate limit (gpt-4o-mini). "
                "Please wait for your quota to reset or raise the limit, "
                "then rerun the analysis."
            )
            return friendly_msg, []
        raise
    task_outputs: List[Tuple[str, str]] = []
    for task_output in results.tasks_output:
        task_outputs.append((str(task_output.agent), task_output.raw))

    final_answer = task_outputs[-1][1] if task_outputs else ""
    return final_answer, task_outputs


if __name__ == "__main__":
    default_image = Path("image1.png")
    default_query = "Summarize the key takeaways from this image."
    answer, outputs = run_image_analysis(default_image, default_query)
    with open("result1.md", "w", encoding="utf-8") as md_file:
        md_file.write("## Crew Execution Results in Markdown:\n\n")
        for idx, (agent_name, raw_output) in enumerate(outputs, start=1):
            md_file.write(f"Agent {idx}: {agent_name}\n{raw_output}\n\n")
        md_file.write("### Final Answer\n")
        md_file.write(answer)
