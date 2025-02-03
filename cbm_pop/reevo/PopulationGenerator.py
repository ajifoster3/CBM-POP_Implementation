import os
import asyncio
from cbm_pop.reevo import reevo_config
from openai import OpenAI


class PopulationGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
        )

    def fetch_function(self, key, function_name, task_description, seed_function, operator_index):
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
                        initial_longterm_reflection="",
                        operator_index=operator_index
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
        population_dict = {}

        # Collect all tasks across all keys
        all_tasks = []

        # Iterate over each key in reevo_config.function_name
        for key in reevo_config.function_name:
            # Prepare parameters for this key
            function_name = reevo_config.function_name[key]
            problem_description = reevo_config.problem_description["task_allocation"]
            function_description = reevo_config.function_description[key]
            seed_function = reevo_config.seed_function[key]
            task_description = reevo_config.prompts["task_description"].format(
                function_name=function_name,
                problem_description=problem_description,
                function_description=function_description,
            )

            # Create population_size tasks for this key
            for i in range(population_size):
                # Create a task that runs fetch_function and tags the result with the key
                task = asyncio.create_task(
                    self._fetch_and_tag(key, function_name, task_description, seed_function, operator_index=i + 1)
                )
                all_tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*all_tasks)

        # Initialize population_dict with empty lists for each key
        population_dict = {key: [] for key in reevo_config.function_name}

        # Populate the results into the dictionary
        for key, result in results:
            population_dict[key].append(result)

        return population_dict

    async def _fetch_and_tag(self, key, function_name, task_description, seed_function, operator_index):
        """Helper to tag results with their key"""
        while True:
            try:
                result = await asyncio.to_thread(
                    self.fetch_function, key, function_name, task_description, seed_function, operator_index
                )
                return key, result
            except Exception as e:
                print(f"Error fetching function for key {key}: {e}")
                return key, ""
