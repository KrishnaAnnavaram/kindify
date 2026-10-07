# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of kindify and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the kindify documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **comment** | One user text that kindify classifies | post, message, input |
| **label** | One of the 6 Jigsaw classes: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` | category, tag |
| **toxicity score** | The probability of the `toxic` label | toxicity (alone), logit, confidence |
| **threshold** | The toxicity score at or above which a comment is toxic | cut-off, limit |
| **natural prevalence** | The real share of toxic comments, with no undersampling | real distribution, base rate (in prose) |
| **classifier** | A model with `predict_proba(texts)`: TF-IDF or transformer | detector, model (alone) |
| **rewriter** | A class with `rewrite(comment)`: rules, OpenAI-compatible or HF chat | paraphraser, generator |
| **rewrite** | The polite text that a rewriter gives | paraphrase, detox output |
| **guard** | One check that a rewrite must pass before anyone sees it | filter, validator |
| **prompt echo** | A rewrite that repeats the system prompt or a few-shot example | prompt leak (in prose), copy |
| **fallback** | The rule rewriter that the service uses when the LLM fails | backup, default model |
| **identity term** | A word from `IDENTITIES`, for example `muslim` or `woman` | group, protected word |
| **subgroup** | The comments that mention one identity term | slice, cohort |
| **BPSN AUC** | AUC on toxic comments without the term and non-toxic comments with it | false-alarm AUC |
| **BNSP AUC** | AUC on toxic comments with the term and non-toxic comments without it | miss AUC |
| **STA** | Share of rewrites below the threshold | success rate, detox rate |
| **SIM** | Character n-gram cosine between comment and rewrite | meaning score, similarity (alone) |
| **feedback store** | The SQLite file with ratings and preference pairs | log, database (alone) |
| **preference pair** | A comment with a chosen and a rejected rewrite | comparison, reward data |
| **run folder** | The folder with `classifier.joblib`, `metrics.json` and `model_card.md` | output, model folder |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **classify** | Give each label a probability for a comment |
| **moderate** | Classify a comment and, if it is toxic, rewrite it with guards |
| **rewrite** | Make a polite text from a comment |
| **guard** | Run the guards on a rewrite |
| **redact** | Replace e-mail addresses, URLs, handles, phone numbers and IP addresses |
| **purge** | Delete feedback rows older than the retention period |
