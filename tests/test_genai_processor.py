from unittest.mock import MagicMock, patch

import pytest
from src.integrations.genai_processor_node import GenAIProcessorNode

# Note: We rely on the fact that the module sets global 'processors = None' if import fails.
# We patch these globals to simulate presence/absence.

@pytest.fixture
def mock_processor_instance():
    # Use MagicMock so we can set an async generator as side_effect
    processor = MagicMock()
    return processor

@pytest.mark.asyncio
async def test_node_initialization(mock_processor_instance):
    # Simulate library PRESENT
    with patch("src.integrations.genai_processor_node.processors", new=MagicMock()):
        node = GenAIProcessorNode(mock_processor_instance)
        assert node.processor == mock_processor_instance

@pytest.mark.asyncio
async def test_node_execution_success(mock_processor_instance):
    # Setup mock behavior for the processor execution
    async def string_generator(input_stream):
        yield MagicMock(text="Chunk 1")
        yield MagicMock(text="Chunk 2")

    mock_processor_instance.side_effect = string_generator

    # Mock the globals in the module to simulate installed library
    with patch("src.integrations.genai_processor_node.processors", new=MagicMock()):
        with patch("src.integrations.genai_processor_node.streams") as mock_streams:
            # Create node
            node = GenAIProcessorNode(mock_processor_instance)

            state = {"content": "Test Input"}
            result = await node(state)

            # Verify usage
            mock_streams.stream_content.assert_called_once()

            # Verify output
            assert result["processed_output"] == "Chunk 1Chunk 2"
            assert result["processor_stats"]["parts_processed"] == 2

@pytest.mark.asyncio
async def test_missing_library_behavior():
    # Simulate library MISSING (force None)
    with patch("src.integrations.genai_processor_node.processors", new=None):
        # Even if we pass an instance, it will warn and set self.processor = None
        # OR we can pass None
        node = GenAIProcessorNode(MagicMock())
        assert node.processor is None

        # Should raise error on execution
        with pytest.raises(ImportError, match="genai-processors not installed"):
            await node({})
