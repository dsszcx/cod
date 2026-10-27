"""Diagnostic prompt variant: replace only the concrete frog JSON example."""
from tools.e0e4 import caption_qwen as Q

PROMPT_VERSION = "e0e4_no_category_example_v2"
SCHEMA = ('{"objects":[{"name":"<category>","sam_text":"<noun phrase>","region_id":1}],'
          '"uncertain":false,"no_visible_object":false}')
_original_build_prompts = Q.build_prompts


def build_prompts(branch):
    system, user = _original_build_prompts(branch)
    if user.count(Q._SCHEMA) != 1:
        raise ValueError("Expected exactly one concrete JSON example")
    return system, user.replace(Q._SCHEMA, SCHEMA)
