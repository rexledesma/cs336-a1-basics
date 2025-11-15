#set enum(numbering: "(a)")
#set heading(numbering: "1.")
#show title: set align(center)
#show title: set block(below: 1.2em)

#title[CS336: Assignment 1]

#align(center)[
  Rex Ledesma \
  Independent \
  #link("mailto:rex.ledesma1@gmail.com")
]

#counter(heading).update(1)
= Byte-Pair Encoding (BPE) Tokenizer
== Understanding Unicode

+ The unicode character returned by `chr(0)` is `'\x00'`, which represents the null character.

+ The string representation is a quoted string of the actual byte, but once it's printed, there is no visual output.

+ When printing the character in the middle of string, it will not be displayed in the output. However, when invoking `repr()`, the null character will be displayed.

== Unicode Encodings

+ As compared to UTF-16 and UTF-32, UTF-8 has many advantages:
  - With UTF-8's variable length encoding (1-4 bytes per character), we have space efficiency in representing our training data.
  - We also have efficiencies when training a BPE representation of our vocabulary using UTF-8 encoding. Rather than wasting learned vocabulary on UTF-32's encoding overhead -- which is especially egregious for ASCII characters that can be respresented in 1 byte in UTF-8 over 4 bytes in UTF-32 -- we can actually learn a semantic compression of our vocabulary in meaningful subwords.
+ The function is incorrect because it incorrectly assumes that only a single byte represents a Unicode codepoint. This assumption does not hold for non-ASCII sequences, like Japanese (e.g. こにちは.)
+ In UTF-8, the leading byte determines allows you to unambiguously identify the subsequent bytes in the byte stream. The leading byte starts with 0 for ASCII characters (1 byte), 110 for 2-byte codes, and 1110 for 3-byte codes. So, to get a byte sequence that does not decode to a Unicode character, we can just choose a byte sequence that does not follow these continuation rules.  For example, `b"\xe1\x80"` terminates too early, as three bytes are expected given the leading byte.

#counter(heading).update((2, 4))
== Experimenting with BPE Tokenizer training

=== BPE Training on TinyStories

+ Running my BPE training algorithm on TinyStories to create a vocabulary of 10000 takes 25.17 seconds wall clock time and \~420 MB max RSS. The longest token is ` accomplishment`, which makes sense as a meaningful word.

+ After profiling with `scalene`, the majority of the time is consumed in the pretokenization step, where we match the regex against corpus of text, and unpack the match as a sequence of bytes.

=== BPE Training on OpenWebText

+ Running my BPE training algorithm on OpenWebText to create a vocabulary of 32000 takes 911.90 seconds wall clock time and 8.20 GB max RSS. The longest token is `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ` which is nonsensical, but occurs multiple times in the text. It looks like an encoding error, which may be a result of encoding errors when the original dataset was scraped from Reddit.

+ Both tokenizers create meaningful subwords. The OpenWebText tokenizer's vocabulary is more diverse, as it includes not only words, but companies (Airbnb), acronyms (CBO), and public figures (Putin). This difference is obviously due to the different training data for each tokenizer.

#counter(heading).update((2, 6))
== Experiments

=== Experiment with Tokenizers

+ For the TinyStories tokenizer, the compression ratio is 4.06 bytes/token. For the OpenWebText tokenizer, the compression ratio is 4.34 bytes/token.

+ When tokenizing the OpenWebText sample with the TinyStories tokenizer, the compression ratio decreases to 3.13. This makes sense for two reasons. First, given the difference in corpuses, the learned tokens from TinyStories may not generalize well to the more diverse dataset of OpenWebText. Second, the TinyStories tokenizer has a smaller vocabulary size, so there is less chance for rarer sequences to be compressed down to a single token.

+ My tokenizer is pretty slow lol. The OpenWebText throughput is 1.6 MB/second. At this rate, it will take 6.1 days to tokenize the pile.

+ When saving the token ids, we need to choose a datatype that can hold the range of token ids that compromise our vocabulary. We need to hold at least 32000 unique token ids, and `uint16`'s max value is 65535, which is satifactory.
