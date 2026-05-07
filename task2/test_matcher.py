import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matcher_engine import get_matcher

matcher = get_matcher()

print("\n" + "="*60)
print("Testing Message Matcher")
print("="*60)

print("\nTest 1: 'What products do you sell?'")
result = matcher.match("What products do you sell?")
print(f"Result keys: {result.keys()}")
print(f"Result: {result}")
answer = matcher.get_answer(result)
print(f"Answer: {answer[:100]}")

print("\nTest 2: 'Je, mnafanya delivery?'")
result = matcher.match("Je, mnafanya delivery?")
print(f"Result keys: {result.keys()}")
print(f"Result: {result}")
answer = matcher.get_answer(result)
print(f"Answer: {answer[:100]}")

print("\nTest 3: 'random test'")
result = matcher.match("random test")
print(f"Result keys: {result.keys()}")
print(f"Result: {result}")
answer = matcher.get_answer(result)
print(f"Answer: {answer[:100]}")

print("\n" + "="*60)
