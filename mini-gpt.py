from utils import device
from model import BigramLanguageModel

m = BigramLanguageModel().to(device)

ckpt: str = "state_dict_3999.pt"
m.load(ckpt)

# m.start_training()

out_len: str = 20  # number of characters to predict
thread: str = ""
while True:
    try:
        prompt: str = input("> ")
        if prompt == "":
            prompt = thread  # continue
        if prompt[0] == "~":
            out_len = int(prompt[1:])
            print(f"[Set response length to {out_len}]")
            continue
        thread = m.evaluate(prompt, out_len=out_len)
        print(thread)
    except KeyboardInterrupt:  # ctrl+c
        print("Goodbye!")
        break
    except EOFError:  # ctrl+D
        print("Goodbye!")
        break
