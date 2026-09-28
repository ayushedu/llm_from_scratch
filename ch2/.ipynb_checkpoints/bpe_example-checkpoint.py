import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "hello, do you like tea?"
print(tokenizer.encode(text))

text2 = "Akwirwier is a word"
print(tokenizer.encode(text2))

ids = tokenizer.encode("Akwirwier")
print([tokenizer.decode([i]) for i in ids])