import pytest

from ateam.base_tool import SharedState


class TestSharedState:
    """Test the SharedState class functionality."""

    def test_init(self):
        """Test SharedState initialization."""
        state = SharedState()
        assert state.data == {}

    def test_set_and_get(self):
        """Test basic set and get operations."""
        state = SharedState()

        # Test setting and getting a value
        state.set("test_key", "test_value")
        assert state.get("test_key") == "test_value"

        # Test getting non-existent key with default
        assert state.get("non_existent", "default") == "default"

        # Test getting non-existent key without default
        assert state.get("non_existent") is None

    def test_set_multiple_values(self):
        """Test setting multiple key-value pairs."""
        state = SharedState()

        state.set("key1", "value1")
        state.set("key2", 42)
        state.set("key3", {"nested": "dict"})

        assert state.get("key1") == "value1"
        assert state.get("key2") == 42
        assert state.get("key3") == {"nested": "dict"}

    def test_overwrite_value(self):
        """Test overwriting existing values."""
        state = SharedState()

        state.set("key", "original")
        assert state.get("key") == "original"

        state.set("key", "updated")
        assert state.get("key") == "updated"

    def test_key_validation(self):
        """Test that keys must be strings."""
        state = SharedState()

        # Test invalid key types for set
        with pytest.raises(ValueError, match="`key` must be a string"):
            state.set(123, "value")

        with pytest.raises(ValueError, match="`key` must be a string"):
            state.set(None, "value")

        # Test invalid key types for get
        with pytest.raises(ValueError, match="`key` must be a string"):
            state.get(123)

        with pytest.raises(ValueError, match="`key` must be a string"):
            state.get(None)

    def test_print_data(self, capsys):
        """Test the print_data method."""
        state = SharedState()
        state.set("key1", "value1")
        state.set("key2", "value2")

        state.print_data()
        captured = capsys.readouterr()

        # Check that both key-value pairs are printed
        assert "key1: value1" in captured.out
        assert "key2: value2" in captured.out
