class LLMUsageCounter:
    def __init__(self):
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def increment_call(self):
        self.call_count += 1

    def update_tokens(self, input_count, output_count):
        self.input_tokens += input_count
        self.output_tokens += output_count
        self.total_tokens = self.input_tokens + self.output_tokens

# Uygulamanın her yerinden erişilebilecek tek bir sayaç örneği
llm_counter = LLMUsageCounter()