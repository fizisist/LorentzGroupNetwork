import re
# Simplify filenames for readability.
# This might not work well with some corner cases,
# needs review for robustness.
def SimplifyPath(path):
    path = re.sub(r'/[^/]*/\.\./', r'/', path)
    if path is re.sub(r'/[^/]*/\.\./', r'/', path):
        return path
    else:
        return SimplifyPath(path)
