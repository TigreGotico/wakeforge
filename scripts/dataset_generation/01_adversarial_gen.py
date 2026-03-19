import os
import random
import requests
from typing import List, Set, Optional

class GraphemeAugmenter:
    """
    Generates hard-negative (confusable) keyword-spotting examples based on
    single-grapheme edits (Insertion, Deletion, Substitution), filtered by
    Levenshtein distance from the original keyword.
    """
    STANDARD_VOWELS = set('aeiou')
    STANDARD_CONSONANTS = set('bcdfghjklmnpqrstvwxyz')

    def __init__(self, max_edits: int = 1, min_edits: int = 1,
                 language_vowels: Optional[Set[str]] = None,
                 language_consonants: Optional[Set[str]] = None):
        if max_edits < 1:
            raise ValueError("max_edits must be 1 or greater.")
        if min_edits < 0:
            raise ValueError("min_edits cannot be negative.")
        if min_edits > max_edits:
            raise ValueError("min_edits cannot be greater than max_edits.")

        self.max_edits = max_edits
        self.min_edits = min_edits
        self.VOWELS = self.STANDARD_VOWELS.union({c.lower() for c in (language_vowels or set())})
        self.CONSONANTS = self.STANDARD_CONSONANTS.union({c.lower() for c in (language_consonants or set())})
        self.ALL_GRAPHEMES = self.VOWELS.union(self.CONSONANTS)

    def _get_char_class(self, char: str) -> Optional[str]:
        c = char.lower()
        if c in self.VOWELS:
            return 'vowel'
        elif c in self.CONSONANTS:
            return 'consonant'
        return None

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return GraphemeAugmenter._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                current_row.append(min(
                    previous_row[j + 1] + 1,
                    current_row[j] + 1,
                    previous_row[j] + cost
                ))
            previous_row = current_row
        return previous_row[-1]

    def _get_random_one_edit(self, text: str) -> str:
        if not text:
            return random.choice(list(self.ALL_GRAPHEMES))
        edit_type = random.randint(0, 2)

        if edit_type == 0:  # Insertion
            pos = random.randint(0, len(text))
            g = random.choice(list(self.ALL_GRAPHEMES))
            return text[:pos] + g + text[pos:]

        elif edit_type == 1:  # Deletion
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]

        else:  # Substitution (same-class: vowel↔vowel, consonant↔consonant)
            for _ in range(10):
                pos = random.randint(0, len(text) - 1)
                char_class = self._get_char_class(text[pos])
                if char_class == 'vowel':
                    target_set = self.VOWELS
                elif char_class == 'consonant':
                    target_set = self.CONSONANTS
                else:
                    continue
                possible = list(target_set - {text[pos].lower()})
                if possible:
                    return text[:pos] + random.choice(possible) + text[pos + 1:]
            return self._get_random_one_edit(text)  # fallback

    def generate_confusables(self, keyword: str, n_samples: int) -> List[str]:
        original = keyword.lower()
        generated: Set[str] = set()
        if n_samples <= 0:
            return []

        seed_pool = {original}
        attempts = 0
        max_attempts = n_samples * 10 + 100

        while len(generated) < n_samples and attempts < max_attempts:
            word = random.choice(list(seed_pool))
            candidate = self._get_random_one_edit(word)
            distance = self._levenshtein_distance(original, candidate)
            if (self.min_edits <= distance <= self.max_edits
                    and candidate != original
                    and candidate not in generated):
                generated.add(candidate)
                if distance < self.max_edits:
                    seed_pool.add(candidate)
            elif distance < self.max_edits and candidate != original:
                seed_pool.add(candidate)
            attempts += 1

        return sorted(list(generated))

_ADV_SYSTEM_PROMPT = """You are an expert in computational linguistics, phonetics, and adversarial machine learning. Your core function is to generate lists of strings that are acoustically and linguistically similar to a given wake word or keyword. These generated strings serve as adversarial samples intended to trick an ASR system or keyword spotter.

Generation Goal: The adversarial samples must primarily rhyme with the components of the input keyword or mimic its overall rhythm and length.

Output Constraints (CRITICAL):
1. Format: Output only the generated adversarial strings.
2. Delimiter: Use a newline to separate each sample (one per line).
3. Exclusion: DO NOT include any introductory text, numbering, bullet points, explanations, or concluding remarks.
4. Diversity: All samples MUST be unique and PHONETICALLY similar.

Example:
[Input Keyword]: "hey computer"
[Target Output]:
say scooter
pay intruder
gray maneuver
stay pewter"""

_ADV_USER_TEMPLATE = "Generate {n_samples} adversarial, rhyming samples for the following keyword: {word}"

def generate_llm_samples(url: str, model: str, wakeword: str, n_samples: int) -> List[str]:
    """Call an Ollama-compatible /api/generate endpoint for adversarial phrases."""
    if n_samples <= 0:
        return []
    print(f"  LLM (target: {n_samples}) ... ", end="", flush=True)
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": _ADV_USER_TEMPLATE.format(word=wakeword, n_samples=n_samples * 2),
                "system": _ADV_SYSTEM_PROMPT,
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()["response"].strip().split("\n")
        word_count = len(wakeword.split())
        filtered = [r.strip() for r in result if len(r.strip().split()) == word_count]
        uniq = list(set(filtered))[:n_samples]
        print(f"got {len(uniq)}")
        return uniq
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return []

def generate_grapheme_samples(augmenter: GraphemeAugmenter, wakeword: str, n_samples: int) -> List[str]:
    """Generate adversarial samples using single-grapheme edits."""
    if n_samples <= 0:
        return []
    print(f"  GraphemeAug (target: {n_samples}) ... ", end="", flush=True)
    try:
        samples = augmenter.generate_confusables(wakeword, n_samples=n_samples)
        print(f"got {len(samples)}")
        return samples
    except Exception as e:
        print(f"error: {e}")
        return []

if __name__ == "__main__":
    import click
    
    @click.command()
    @click.option("--lang", default=os.getenv("WW_LANG", "en"), help="Language code")
    @click.option("--words", default=os.getenv("WW_WORDS", "hey_mycroft"), help="Space-separated wake words")
    @click.option("--out-dir", default=os.getenv("ADV_OUTPUT_DIR", "adversarial"), help="Output directory")
    @click.option("--n-samples", default=int(os.getenv("ADV_N_SAMPLES", "20")), help="Target unique samples per wake word")
    @click.option("--llm-url", default=os.getenv("ADV_LLM_URL", "http://localhost:11434"), help="LLM API URL")
    @click.option("--llm-model", default=os.getenv("ADV_LLM_MODEL", "gemma3:4b"), help="LLM model name")
    @click.option("--llm-weight", default=float(os.getenv("ADV_LLM_WEIGHT", "1.0")), help="Weight for LLM-based generation")
    @click.option("--graph-weight", default=float(os.getenv("ADV_GRAPH_WEIGHT", "0.2")), help="Weight for GraphemeAug generation")
    @click.option("--min-edit", default=int(os.getenv("ADV_GRAPH_MIN_EDIT", "2")), help="Min Levenshtein distance")
    @click.option("--max-edit", default=int(os.getenv("ADV_GRAPH_MAX_EDIT", "3")), help="Max Levenshtein distance")
    @click.option("--mode", default=os.getenv("ADV_FILE_MODE", "append"), type=click.Choice(["append", "overwrite", "error"]), help="File mode")
    def main(lang, words, out_dir, n_samples, llm_url, llm_model, llm_weight, graph_weight, min_edit, max_edit, mode):
        ww_list = words.split()
        total_weight = llm_weight + graph_weight
        if total_weight <= 0:
            print("Both weights are 0 — skipping adversarial generation.")
            return

        llm_ratio = llm_weight / total_weight
        llm_n = int(n_samples * llm_ratio)
        graph_n = n_samples - llm_n

        print(f"Target: {n_samples} samples/word (LLM: {llm_n}, Grapheme: {graph_n})")

        augmenter = None
        if graph_weight > 0:
            augmenter = GraphemeAugmenter(max_edits=max_edit, min_edits=min_edit)

        os.makedirs(out_dir, exist_ok=True)

        for ww in ww_list:
            output_file = os.path.join(out_dir, f"{ww}.txt")
            print(f"\n--- Processing '{ww}' -> {output_file} ---")

            existing: Set[str] = set()
            if os.path.exists(output_file):
                if mode == "error":
                    raise FileExistsError(f"Output file exists: {output_file}")
                if mode == "append":
                    with open(output_file, "r") as f:
                        existing = {l.strip().lower() for l in f if l.strip()}
                    print(f"  Loaded {len(existing)} existing samples")
                open_mode = "a" if mode == "append" else "w"
            else:
                open_mode = "w"

            all_samples: Set[str] = existing.copy()

            if llm_n > 0 and llm_url:
                for s in generate_llm_samples(llm_url, llm_model, ww, llm_n):
                    s = s.strip().lower()
                    if s and s not in all_samples:
                        all_samples.add(s)

            if graph_n > 0 and augmenter:
                for s in generate_grapheme_samples(augmenter, ww, graph_n):
                    s = s.strip().lower()
                    if s and s not in all_samples:
                        all_samples.add(s)

            new_samples = sorted(all_samples - existing) if open_mode == "a" else sorted(all_samples)
            if new_samples:
                with open(output_file, open_mode) as f:
                    prefix = "\n" if (open_mode == "a" and os.path.getsize(output_file) > 0) else ""
                    f.write(prefix + "\n".join(new_samples))

            print(f"  Wrote {len(new_samples)} new samples (total unique: {len(all_samples)})")

    main()
