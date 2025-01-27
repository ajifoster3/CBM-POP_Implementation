import os
import asyncio
from cbm_pop.reevo import reevo_config
from openai import OpenAI


class PopulationGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
            project='proj_bMwxaxkpiKADKECQtHOCCUjh',
        )

    def fetch_function(self, key, function_name, task_description, seed_function):
        function = ""
        # Synchronous call to OpenAI API
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": reevo_config.prompts["generator_system_prompt"]
                },
                {
                    "role": "user",
                    "content": reevo_config.prompts["user_prompt_population_initialisation"].format(
                        task_description=task_description,
                        seed_function=seed_function,
                        function_name=function_name,
                        initial_longterm_reflection=""
                    )
                }
            ],
            stream=True,
        )

        # Handle the streaming response synchronously (blocking call)
        function = self.stream_response(stream)
        return function

    def stream_response(self, stream):
        """Helper function to handle the stream in a synchronous manner."""
        function = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                function += content  # Append the content to the `function` string
        return function

    async def generate_population(self, population_size):
        population = {}

        # Prepare tasks for all keys in reevo_config
        tasks = []

        key = reevo_config.function_name["ga_combined"]

        function_name = reevo_config.function_name[key]
        problem_description = reevo_config.problem_description["task_allocation"]
        function_description = reevo_config.function_description[key]
        seed_function = reevo_config.seed_function[key]
        task_description = reevo_config.prompts["task_description"].format(
            function_name=function_name,
            problem_description=problem_description,
            function_description=function_description,
        )
        # Create async tasks for generating functions, using asyncio.to_thread to offload blocking calls
        for i in range(population_size):
            tasks.append(asyncio.to_thread(self.fetch_function, key, function_name, task_description, seed_function))

        # Run all tasks concurrently
        population = await asyncio.gather(*tasks)

        return population
