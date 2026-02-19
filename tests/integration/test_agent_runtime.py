"""Integration tests for the Agent runtime.

Tests the ReAct loop with real tool execution (CalculatorTool,
SearchTool backed by real DB), multi-step reasoning, error recovery,
and max-steps stopping. LLM responses are mocked; tools execute for real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from src.agent.runtime import AgentRuntime
from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.registry import ToolRegistry
from src.agent.tools.search import SearchTool
from src.db.models import Document, User
from src.db.vector_store import VectorStore
from src.llm.providers.base import LLMResponse, TokenUsage
from src.rag.chunker import RecursiveCharacterSplitter
from src.rag.generator import Generator
from src.rag.index_manager import IndexManager
from src.rag.pipeline import RAGPipeline
from src.rag.preprocessor import Preprocessor
from src.rag.retriever import Retriever

from .conftest import FakeEmbedder, make_mock_llm_provider

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _make_llm_response(content: str) -> LLMResponse:
    """Helper to build an LLMResponse."""
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


# ===========================================================================
# TestAgentWithCalculatorTool
# ===========================================================================
class TestAgentWithCalculatorTool:
    """Tests for agent using real CalculatorTool execution."""

    async def test_agent_uses_calculator_and_returns_correct_answer(self) -> None:
        """Agent calls calculator with 6*7, gets observation '42', returns final answer."""
        provider = AsyncMock()
        provider.complete.side_effect = [
            # Step 1: LLM decides to use calculator
            _make_llm_response(
                "Thought: I need to calculate 6*7\nAction: calculator\nAction Input: 6*7"
            ),
            # Step 2: LLM sees observation "42" and gives final answer
            _make_llm_response(
                "Thought: The calculator returned 42\nFinal Answer: 6 times 7 equals 42"
            ),
        ]

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
        )

        result = await runtime.run("What is 6 times 7?")

        assert result.answer == "6 times 7 equals 42"
        assert result.total_steps == 2
        # First step should have used calculator with real execution
        assert result.steps[0].action == "calculator"
        assert result.steps[0].observation == "42"
        assert result.steps[0].is_error is False

    async def test_agent_handles_calculator_division_by_zero(self) -> None:
        """Division by zero produces an error observation, then agent gives final answer."""
        provider = AsyncMock()
        provider.complete.side_effect = [
            _make_llm_response(
                "Thought: I need to calculate 1/0\nAction: calculator\nAction Input: 1/0"
            ),
            _make_llm_response(
                "Thought: Division by zero error occurred\nFinal Answer: Cannot divide by zero"
            ),
        ]

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
        )

        result = await runtime.run("What is 1 divided by 0?")

        assert result.steps[0].is_error is True
        assert result.steps[0].observation is not None
        assert "division by zero" in result.steps[0].observation.lower()
        assert result.answer == "Cannot divide by zero"

    async def test_agent_multi_step_calculation(self) -> None:
        """Agent performs multiple calculator calls in succession."""
        provider = AsyncMock()
        provider.complete.side_effect = [
            _make_llm_response(
                "Thought: First calculate 10 + 20\nAction: calculator\nAction Input: 10 + 20"
            ),
            _make_llm_response(
                "Thought: Now multiply the result by 3\nAction: calculator\nAction Input: 30 * 3"
            ),
            _make_llm_response("Thought: The final result is 90\nFinal Answer: The result is 90"),
        ]

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
        )

        result = await runtime.run("Calculate (10+20)*3")

        assert result.total_steps == 3
        assert result.steps[0].observation == "30"
        assert result.steps[1].observation == "90"
        assert result.answer == "The result is 90"


# ===========================================================================
# TestAgentWithSearchTool
# ===========================================================================
class TestAgentWithSearchTool:
    """Tests for agent using SearchTool backed by real DB."""

    async def test_agent_search_tool_queries_real_database(
        self,
        db_session: SQLModelAsyncSession,
        test_user: User,
    ) -> None:
        """Agent retrieves indexed documents from real DB via SearchTool."""
        # Index a document
        embedder = FakeEmbedder()
        vector_store = VectorStore(session=db_session)
        preprocessor = Preprocessor()
        chunker = RecursiveCharacterSplitter(chunk_size=128, chunk_overlap=16)
        index_manager = IndexManager(
            preprocessor=preprocessor,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        retriever = Retriever(embedder=embedder, vector_store=vector_store)
        gen_provider = make_mock_llm_provider(content="Python is a programming language")
        generator = Generator(llm_provider=gen_provider, model="test-model")
        pipeline = RAGPipeline(
            index_manager=index_manager,
            retriever=retriever,
            generator=generator,
        )

        doc = Document(
            title="Python Guide",
            content="Python is a high-level programming language. " * 10,
            user_id=test_user.id,
        )
        db_session.add(doc)
        await db_session.flush()
        await pipeline.index_document(doc)

        # Set up agent with SearchTool
        search_tool = SearchTool(pipeline=pipeline, top_k=3)
        registry = ToolRegistry()
        registry.register(search_tool)

        agent_provider = AsyncMock()
        agent_provider.complete.side_effect = [
            _make_llm_response(
                "Thought: I should search for Python info\n"
                "Action: search\n"
                "Action Input: What is Python?"
            ),
            _make_llm_response(
                "Thought: I found the information\nFinal Answer: Python is a programming language"
            ),
        ]

        runtime = AgentRuntime(
            llm_provider=agent_provider,
            model="test-model",
            tool_registry=registry,
        )

        result = await runtime.run("Tell me about Python")

        assert result.total_steps == 2
        assert result.steps[0].action == "search"
        assert result.steps[0].is_error is False
        # The observation should contain the search result from the real DB
        assert result.steps[0].observation is not None
        assert "Python" in result.steps[0].observation


# ===========================================================================
# TestAgentMaxStepsIntegration
# ===========================================================================
class TestAgentMaxStepsIntegration:
    """Tests for agent max-steps stopping with real tools."""

    async def test_agent_stops_at_max_steps_with_real_tools(self) -> None:
        """Agent stops when max_steps is reached and sets stopped_by_max_steps."""
        provider = AsyncMock()
        # Always return a calculator action — never gives a final answer
        provider.complete.return_value = _make_llm_response(
            "Thought: I need to keep calculating\nAction: calculator\nAction Input: 1+1"
        )

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
            max_steps=3,
        )

        result = await runtime.run("Keep calculating")

        assert result.stopped_by_max_steps is True
        assert result.total_steps == 3


# ===========================================================================
# TestAgentErrorRecovery
# ===========================================================================
class TestAgentErrorRecovery:
    """Tests for agent recovering from errors and continuing."""

    async def test_agent_recovers_from_unknown_tool_and_uses_calculator(self) -> None:
        """Agent tries unknown tool (error), then uses calculator successfully."""
        provider = AsyncMock()
        provider.complete.side_effect = [
            # Step 1: Try a non-existent tool
            _make_llm_response(
                "Thought: Let me use the weather tool\nAction: weather\nAction Input: Tokyo"
            ),
            # Step 2: Recover and use calculator
            _make_llm_response(
                "Thought: Weather tool not found, let me use calculator\n"
                "Action: calculator\n"
                "Action Input: 2 + 2"
            ),
            # Step 3: Final answer
            _make_llm_response("Thought: Got the calculation result\nFinal Answer: 2 plus 2 is 4"),
        ]

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        runtime = AgentRuntime(
            llm_provider=provider,
            model="test-model",
            tool_registry=registry,
        )

        result = await runtime.run("What is 2+2?")

        # Step 1 had an error (unknown tool)
        assert result.steps[0].is_error is True
        assert result.steps[0].observation is not None
        assert "not found" in result.steps[0].observation.lower()
        # Step 2 succeeded with calculator
        assert result.steps[1].action == "calculator"
        assert result.steps[1].observation == "4"
        assert result.steps[1].is_error is False
        # Final answer
        assert result.answer == "2 plus 2 is 4"
