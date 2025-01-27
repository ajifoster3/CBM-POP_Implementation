import os
from openai import OpenAI
from cbm_pop.reevo import reevo_config


class ShortTermReflector:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
            project='proj_bMwxaxkpiKADKECQtHOCCUjh',
        )
        pass

    def fetch_reflection(self,
                         function_name,
                         problem_description,
                         function_description,
                         worse_code,
                         better_code):
        function = ""
        # Synchronous call to OpenAI API
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": reevo_config.prompts["reflector_system_prompt"]
                },
                {
                    "role": "user",
                    "content": reevo_config.prompts["user_prompt_shortterm_reflection"].format(
                        function_name=function_name,
                        problem_description=problem_description,
                        function_description=function_description,
                        worse_code=worse_code,
                        better_code=better_code
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