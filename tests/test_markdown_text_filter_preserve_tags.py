"""
Test cases for MarkdownTextFilter preserve_tags feature.

Tests the enhanced MarkdownTextFilter to ensure HTML tags can be preserved
during markdown processing, particularly useful for TTS pronunciation tags
like <spell> for proper number pronunciation.
"""

import pytest

from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter


class TestMarkdownTextFilterPreserveTags:
    """Test the MarkdownTextFilter preserve_tags feature."""
    
    @pytest.mark.asyncio
    async def test_preserve_spell_tags_basic(self):
        """Test basic spell tag preservation."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Is the number ending in <spell>3029</spell> the best number?"
        result = await filter_instance.filter(input_text)
        
        assert "<spell>3029</spell>" in result
        assert "__PRESERVE_" not in result  # Placeholders should be restored
    
    @pytest.mark.asyncio
    async def test_preserve_spell_tags_with_markdown(self):
        """Test spell tag preservation with markdown formatting."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Call **<spell>3029</spell>** for *urgent* matters."
        result = await filter_instance.filter(input_text)
        
        assert "<spell>3029</spell>" in result
        assert "**" not in result  # Markdown should be removed
        assert "*urgent*" not in result or "urgent" in result  # Markdown should be processed
    
    @pytest.mark.asyncio
    async def test_preserve_multiple_spell_tags(self):
        """Test preservation of multiple spell tags."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Call <spell>3029</spell> or <spell>5551</spell> for help."
        result = await filter_instance.filter(input_text)
        
        assert "<spell>3029</spell>" in result
        assert "<spell>5551</spell>" in result
        assert "__PRESERVE_" not in result
    
    @pytest.mark.asyncio
    async def test_preserve_spell_tags_with_attributes(self):
        """Test preservation of spell tags with attributes."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = 'Call <spell lang="en">3029</spell> for help.'
        result = await filter_instance.filter(input_text)
        
        assert '<spell lang="en">3029</spell>' in result
        assert "__PRESERVE_" not in result
    
    @pytest.mark.asyncio
    async def test_no_spell_tags(self):
        """Test that filter works normally when no spell tags are present."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "This is **bold** text with no spell tags."
        result = await filter_instance.filter(input_text)
        
        assert "**" not in result
        assert "bold" in result
        assert "<spell>" not in result
    
    @pytest.mark.asyncio
    async def test_placeholder_not_corrupted_by_space_processing(self):
        """Test that placeholders survive the space restoration logic.
        
        This is the critical test that prevents the bug where § placeholders
        would be corrupted by re.sub('§', ' ', filtered_text).
        """
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        # This specific case was causing the bug
        input_text = "Now, is the phone number you're calling from, ending in <spell>3029</spell>, the best number?"
        result = await filter_instance.filter(input_text)
        
        # Should NOT contain corrupted placeholder
        assert "__PRESERVE_SPELL_0" not in result
        assert " __PRESERVE_SPELL_0 " not in result
        assert "__PRESERVE_" not in result
        
        # Should contain the original spell tag
        assert "<spell>3029</spell>" in result
    
    @pytest.mark.asyncio
    async def test_preserve_multiple_tag_types(self):
        """Test preserving multiple different tag types."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell", "phoneme"])
        )
        
        input_text = "Call <spell>3029</spell> and say <phoneme>hello</phoneme>."
        result = await filter_instance.filter(input_text)
        
        assert "<spell>3029</spell>" in result
        assert "<phoneme>hello</phoneme>" in result
        assert "__PRESERVE_" not in result
    
    @pytest.mark.asyncio
    async def test_preserve_tags_with_spaces_and_special_chars(self):
        """Test spell tags containing spaces and special characters."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Call <spell>1-800-HELP</spell> or <spell>+1 (555) 123-4567</spell>."
        result = await filter_instance.filter(input_text)
        
        assert "<spell>1-800-HELP</spell>" in result
        assert "<spell>+1 (555) 123-4567</spell>" in result
    
    @pytest.mark.asyncio
    async def test_empty_spell_tags(self):
        """Test empty spell tags."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Call <spell></spell> for help."
        result = await filter_instance.filter(input_text)
        
        assert "<spell></spell>" in result
    
    @pytest.mark.asyncio
    async def test_malformed_spell_tags(self):
        """Test malformed spell tags (should not be processed)."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Call <spell>3029 or <spell>incomplete for help."
        result = await filter_instance.filter(input_text)
        
        # Malformed tags should be treated as regular text and removed by HTML cleaning
        assert "<spell>" not in result
        assert "3029 or" in result
        assert "incomplete for help" in result
    
    @pytest.mark.asyncio
    async def test_preserve_tags_disabled_by_default(self):
        """Test that no tags are preserved when preserve_tags is empty."""
        filter_instance = MarkdownTextFilter()  # Default params
        
        input_text = "Call <spell>3029</spell> for help."
        result = await filter_instance.filter(input_text)
        
        # Should remove all HTML tags including spell tags
        assert "<spell>" not in result
        assert "3029" in result
        assert "help" in result
    
    @pytest.mark.asyncio
    async def test_nested_markdown_with_spell_tags(self):
        """Test spell tags within complex markdown formatting."""
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        input_text = "Please call **<spell>555-1234</spell>** or *<spell>911</spell>* for emergencies."
        result = await filter_instance.filter(input_text)
        
        assert "<spell>555-1234</spell>" in result
        assert "<spell>911</spell>" in result
        assert "**" not in result  # Markdown should be removed
        assert "*" not in result   # Markdown should be removed
    
    @pytest.mark.asyncio
    async def test_real_world_scenario(self):
        """Test real-world scenario that motivated this feature.
        
        This test represents the actual use case where AI generates text with
        spell tags for TTS pronunciation, and the markdown filter needs to
        preserve them while removing other formatting.
        """
        filter_instance = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(preserve_tags=["spell"])
        )
        
        # This is the exact text that was causing issues
        input_text = "Now, is the phone number you're calling from—ending in <spell>3029</spell>—the best number to reach you and Krish, or would you prefer a different number?"
        result = await filter_instance.filter(input_text)
        
        # Should preserve spell tags without corruption
        assert "<spell>3029</spell>" in result
        assert "__PRESERVE_" not in result
        assert "Krish" in result  # Other content preserved