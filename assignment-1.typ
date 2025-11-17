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

= Transformer Language Model Architecture

#counter(heading).update((3, 5))
== The Full Transformer Language Model

=== Transformer Language Model Resource Accounting

#let d_vocab = 50257
#let context_length = 1024
#let n_layers = 48
#let d_model = 1600
#let n_heads = 25
#let d_ff = 6400

+ GPT-2 XL has the following parameters:
  - $d_"vocab" = #{ d_vocab }$
  - $L = #{ context_length }$
  - $n_"layers"$: #{ n_layers }
  - $d_"model"$: #{ d_model }
  - $n_"heads"$: #{ n_heads }
  - $d_"ff": #{ d_ff }$

  To calculate the number of parameters in the model given this configuration, we need to substitute all these values in our current model.
  Alternatively, we can just instantiate this model and get the number of parameters from it.

  #let p_embedding = 2 * d_vocab * d_model
  #let p_rms = (2 * n_layers + 1) * d_model
  #let p_attention = 4 * n_layers * calc.pow(d_model, 2)
  #let p_fnn = 3 * n_layers * d_model * d_ff
  #let p_total = p_embedding + p_rms + p_attention + p_fnn
  #table(
    columns: 4,
    table.header[*Layer*][*Parameters*][*Actual*][*Percentage*],
    [Embedding + De-embedding],
    [2 $d_"vocab" d_"model"$],
    [#{ p_embedding }],
    [#{ calc.round(p_embedding / p_total, digits: 3) * 100 }%],

    [RMS Norm], [$(2 n_"layers" + 1) d_"model"$], [#{ p_rms }], [#{ calc.round(p_rms / p_total, digits: 3) * 100 }%],

    [Attention],
    [$4 n_"layers" d_"model"^2$],
    [#{ p_attention }],
    [#{ calc.round(p_attention / p_total, digits: 3) * 100 }%],

    [Feedforward],
    [$3 n_"layers" d_"model" d_"ff"$],
    [#{ p_fnn }],
    [#{ calc.round(p_fnn / p_total, digits: 3) * 100 }%],

    [*Total*], [], [*#{ p_total }*], [],
  )

  So using GPT-2 XL's configuration, we construct a model with about 2.1B parameters.

  Assuming that we're using `float32`, we'd need about 8.4 GB of memory to load the model parameters (4 bytes per parameter).

+ The matrix multiplies in the transformer model are found in the attention and feed-forward network layers.

  #let transformer_flops(context_length, n_layers, d_model, d_ff, caption) = {
    let attention_projection_flops = 4 * 2 * n_layers * context_length * calc.pow(d_model, 2)
    let attention_flops = 4 * n_layers * calc.pow(context_length, 2) * d_model
    let fnn_flops = 6 * n_layers * context_length * d_model * d_ff
    let total_flops = attention_projection_flops + attention_flops + fnn_flops

    figure(
      table(
        columns: 4,
        table.header[*Operation*][*FLOPs *][*Actual*][*Percentage*],
        [Attention Projections],
        [$n_"layers" (8 L d_"model"^2)$],
        [#{ attention_projection_flops }],
        [#{ calc.round(attention_projection_flops / total_flops, digits: 4) * 100 }%],

        [Attention],
        [$n_"layers" (2 L^2 d_"model")$],
        [#{ attention_flops }],
        [#{ calc.round(attention_flops / total_flops, digits: 4) * 100 }%],

        [Feedforward],
        [$n_"layers" (6 L d_"model" d_"ff")$],
        [#{ fnn_flops }],
        [#{ calc.round(fnn_flops / total_flops, digits: 4) * 100 }%],

        [*Total*], [], [#{ total_flops }], [],
      ),
      caption: caption + " (" + ($n_"layers" = #n_layers, d_"model" = #d_model, d_"ff" = #d_ff$) + ")",
    )
  }

  #transformer_flops(context_length, n_layers, d_model, d_ff, "GPT-2 XL")

+ With the GPT-2 XL configuration, most of the FLOPs come from the feedforward layer and the projection of the input to the QKV space.

+ As the model dimension increases, most of the FLOPs come from the feedforward layers, as the feedforward layer is proportional to the square of the model dimension (assuming that $d_"ff" = 4 d_"model"$).

  #{
    let analyses = (("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024), ("GPT-2 large", 36, 1280))

    for (caption, n_layers, d_model) in analyses {
      let d_ff = 4 * d_model

      transformer_flops(
        1024,
        n_layers,
        d_model,
        d_ff,
        caption,
      )
    }
  }

+ #transformer_flops(16 * context_length, n_layers, d_model, d_ff, "GPT-2 XL, L=16384")

  As context length increases, more FLOPs come from the attention operation. This is because the attention FLOPs are proportional to the square of context length.

= Training a Transformer Language Model

#counter(heading).update((4, 1))
== The SGD Optimizer

=== Tuning the Learning Rate

At a learning rate of 1e1, the loss decays too slow. Meanwhile, a learning rate of 1e2 causes the loss to converge to 0, while a learning rate of 1e3 causes the loss to diverge.
