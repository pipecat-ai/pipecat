"""Mock module for datetime to allow patching datetime.now().

Since datetime.datetime is immutable in Python, we can't directly patch datetime.now.
Instead, we create a wrapper around the real datetime module that we can patch.
"""

import datetime
from unittest.mock import MagicMock

# Keep a reference to the real datetime class
real_datetime = datetime.datetime


class MockDatetime(real_datetime):
    """A datetime replacement that can be mocked for testing."""
    
    @classmethod
    def now(cls, tz=None):
        """Return the mocked current date and time."""
        return cls._now_value
    
    @classmethod
    def set_now(cls, value):
        """Set the value that now() will return."""
        cls._now_value = value


# Default now value
MockDatetime._now_value = real_datetime(2023, 1, 1, 12, 0, 0)


def patch_datetime():
    """Replace the real datetime with our mock version for tests."""
    datetime.datetime = MockDatetime
    return MockDatetime


def unpatch_datetime():
    """Restore the original datetime module."""
    datetime.datetime = real_datetime 