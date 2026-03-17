"""
GSM8K 4-shot Chain-of-Thought prompt used in Qwen paper evaluation.

The prompt format follows the standard GSM8K evaluation used by
OpenAI / DeepMind math benchmarks.

We include 4 solved examples with reasoning traces.

The model will generate reasoning + final answer.
"""

FEW_SHOT_PROMPT = """
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today.
After they are done, there will be 21 trees. How many trees did the workers plant today?
Answer:
There are 15 trees originally. After planting there are 21 trees.
So the workers planted 21 - 15 = 6 trees.
The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive,
how many cars are in the parking lot?
Answer:
There are originally 3 cars. 2 more arrive.
So total cars = 3 + 2 = 5.
The answer is 5.

Question: Leah had 32 chocolates and her sister had 42.
If they ate 35, how many pieces do they have left in total?
Answer:
Total chocolates initially = 32 + 42 = 74.
They ate 35.
Remaining = 74 - 35 = 39.
The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some.
Now Jason has 12 lollipops. How many did he give to Denny?
Answer:
Jason started with 20.
Now he has 12.
So he gave away 20 - 12 = 8.
The answer is 8.

Question: {question}
Answer:
"""