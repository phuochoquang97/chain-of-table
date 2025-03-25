# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from groq import Groq
import time
import numpy as np


class ChatGPT:
    def __init__(self, model_name, key):
        self.model_name = model_name
        self.key = key
        self.client = Groq( api_key=self.key)
        

    def get_model_options(
        self,
        temperature=0,
        per_example_max_decode_steps=150,
        per_example_top_p=1,
        n_sample=1,
    ):
        return dict(
            temperature=temperature,
            n=n_sample,
            top_p=per_example_top_p,
            max_completion_tokens=per_example_max_decode_steps,
        )

    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        messages = [
            {
                "role": "system",
                "content": "I will give you some examples, you need to follow the examples and complete the text, and no other content.",
            },
            {"role": "user", "content": prompt},
        ]
        gpt_responses = None
        retry_num = 0
        retry_limit = 2
        error = None
        while gpt_responses is None:
            try:
                gpt_responses = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=np.array(messages),
                    temperature=options['temperature'],
                    max_completion_tokens=options['max_completion_tokens'],
                    top_p=1,
                    stop=end_str,
                )
                
                error = None
            except Exception as e:
                print(f"Error: {str(e)}", flush=True)
                error = str(e)
                retry_num += 1
                time.sleep(10)  # Reduced wait time
        if error:
            raise Exception(error)
        
        if hasattr(gpt_responses, "choices") and gpt_responses.choices:
            text = gpt_responses.choices[0].message.content or ""
        else:
            raise ValueError("Invalid API response: 'choices' field missing or empty")

        return [(text, np.float64(0.0))]

    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)[0][0]
        return result
