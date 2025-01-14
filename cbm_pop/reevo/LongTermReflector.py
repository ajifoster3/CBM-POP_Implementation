import os
from openai import OpenAI

from cbm_pop.reevo import reevo_config


class LongTermReflector:
    def __init__(self):
        self.prior_longterm_reflection = ""
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
            project='proj_bMwxaxkpiKADKECQtHOCCUjh',
        )
        pass

    def perform_longterm_reflection(self,
                                    new_shortterm_reflection,
                                    problem_description ):
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
                    "content": reevo_config.prompts["user_prompt_longterm_reflection"].format(
                        prior_longterm_reflection=self.prior_longterm_reflection,
                        new_shortterm_reflection=new_shortterm_reflection,
                        problem_description=problem_description,
                    )
                }
            ],
            stream=True,
        )

        # Handle the streaming response synchronously (blocking call)
        function = self.stream_response(stream)
        self.prior_longterm_reflection = function
        return function

    def stream_response(self, stream):
        """Helper function to handle the stream in a synchronous manner."""
        function = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                function += content  # Append the content to the `function` string
        return function
