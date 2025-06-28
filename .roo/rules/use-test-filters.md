# Use Test Filters to Isolate Examples

When fixing a specific example's dependencies or imports, use the filtering feature of `test_all_examples.py` to speed up the feedback loop.

**Command:**
`python test_all_examples.py [substring_of_path_or_content]`

By providing a unique substring from the file's path or its content, you can instruct the script to run tests only on the relevant files, avoiding the time-consuming process of setting up environments for unrelated examples.