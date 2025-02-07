import json
import base64
from collections import Counter

class BasicTokenizer:
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        self.mint_trees = {}
 
        for idx in range(255, vocab_size):
            pair, _ = get_stats(tokens)
            idx += 1
            self.mint_trees[pair] = idx
            tokens = mint_token_and_merge(tokens, idx, pair)
            if verbose:
                print(f"{pair} is merged into {idx}")
        self._get_docoder_map()
        self._save()

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while True:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.mint_trees:
                    new_tokens.append(self.mint_trees[(tokens[i], tokens[i + 1])])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if len(tokens) == len(new_tokens):
                break
            tokens = new_tokens[:]
        
        return tokens

    def decode(self, ids):
        res = b"".join([self.decoder_map[el] for el in ids])
        return bytes(res).decode("utf-8", errors="replace")
    
    def _get_docoder_map(self):
        self.decoder_map = {i:bytes([i]) for i in range(256)}
        for (idx1, idx2), key in self.mint_trees.items():
            self.decoder_map[key] = self.decoder_map[idx1] + self.decoder_map[idx2]

    def _save(self):
        decoder_map_data = {k: bytes(v).decode('utf-8', errors="replace") for k, v in self.decoder_map.items()}
        # Save to JSON
        with open("decoder_map.json", "w") as f:
            json.dump(decoder_map_data, f, indent=4)

        merger_data = {str(k): v for k, v in self.mint_trees.items()}
        with open("merger.json", "w") as f:
            json.dump(merger_data, f, indent=4)
        

def get_stats(tokens):
    pairs = zip(tokens[:-1], tokens[1:])
    count = Counter(pairs)
    return max(count, key=count.get), count

def mint_token_and_merge(tokens, idx, pair):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    # text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."""
    with open("test/taylorswift.txt", "r") as file:
        text = file.read()
    print(text)

    vocab_size = 1000
    tokenizer.train(text, vocab_size, verbose=False)
    # print(tokenizer.decoder_map)

    sample_text = "Many common characters, including numerals, punctuation, and other symbols,"
    encoded_int = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_int)
    print(encoded_int)
    print(decoded_text)

