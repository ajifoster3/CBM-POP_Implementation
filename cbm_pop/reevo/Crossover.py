import asyncio
import os
from openai import OpenAI

from cbm_pop.reevo import reevo_config


class Crossover:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
            project='proj_bMwxaxkpiKADKECQtHOCCUjh',
        )
        pass

    def perform_crossover(self,
                          function_name,
                          task_description,
                          function_signature0,
                          worse_code,
                          function_signature1,
                          better_code,
                          shortterm_reflection):
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
                    "content": reevo_config.prompts["user_prompt_crossover"].format(
                        function_name=function_name,
                        task_description=task_description,
                        function_signature0=function_signature0,
                        worse_code=worse_code,
                        function_signature1=function_signature1,
                        better_code=better_code,
                        shortterm_reflection=shortterm_reflection
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
