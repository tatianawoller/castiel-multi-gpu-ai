# 1A: Introduction to Deep Learning


```{toctree}
:maxdepth: 1
:caption: Episodes

setup
1-introduction
2-keras
3-monitor-the-model
```

## Code and Jupyter notebooks

````{note}
The files are added here
{{repo}}/se/deep-learning-intro/notebooks

Access them by doing:
```sh
git clone https://github.com/ENCCS/castiel-multi-gpu-ai
cd castiel-multi-gpu-ai/content/se/deep-learning-intro/notebooks
```

````

To launch jupyter start with {ref}`instructions here <jupyter>` and then use the {download}`notebooks/start_jupyter_for_deep_learning_intro.sh` shown below:

```{warning}
To avoid wasting resources, remember to **save** (using the Jupyter interface) and **cancel** (using `scancel --me`) the jobs at the end of the session.
```

```{literalinclude} notebooks/start_jupyter_for_deep_learning_intro.sh
:language: bash
```


- {download}`notebooks/1_DL_test.py`
- {download}`notebooks/2-Classification-NN-Keras-PenguinsClassification.ipynb`
- {download}`notebooks/3-Monitor-training-process-WeatherPrediction.ipynb`

<!-- 
:::{admonition} Extra reading material
:class: dropdown

```{toctree}
:maxdepth: 1

4-advanced-layer-types
5-transfer-learning
6-outlook
```

:::
 -->


```{toctree}
:maxdepth: 1
:caption: Reference

reference
```

## Schedule

All times in CET.


| Time | Topic | Instructor |
| ------ | ------- | ------- |
| 14:00 | [Introduction to Deep Learning](./1-introduction.md) | YW |
| 14:15 | [Classification by a neural network using Keras (§ 1-6)](./2-keras.md) | YW |
| 15:15 | Exercises (10 min) |  |
| 15:25 | [Classification by a neural network using Keras (§ 7-10)](./2-keras.md#perform-a-prediction-classification) | YW |
| 15:45 | Coffee Break (15 min) |  |
| 16:00 | [Monitor the training process (§ 1-6)](./3-monitor-the-model.md) | AM |
| 16:35 | Exercises (10 min) |  |
| 16:45 | [Monitor the training process (§ 7-10)](./3-monitor-the-model.md#perform-a-prediction-classification) | AM |
| 17:15 | Exercises (15 min) |  |
| 17:30 | End of day 1 | |

## Other related lessons

### Original carpentries lesson

This lesson is an adaptation of

- <https://carpentries-lab.github.io/deep-learning-intro>
- <https://enccs.github.io/deep-learning-intro/>
- <https://mimer-ai.github.io/deep-learning-intro/>


all licensed CC-BY-4.0 

```{include} ../index.md
:start-after: Other related lessons
:end-before: We can help you out
```

::::{admonition} License
:class: attention

:::{admonition} CC BY for media and pedagogical material
:class: attention dropdown

Copyright © 2025 Carpentries. This material is released by Carpentries (Software Carpentry, Data Carpentry, and Library Carpentry) under the Creative Commons Attribution 4.0 International (CC BY 4.0).

**Canonical URL**: <https://creativecommons.org/licenses/by/4.0/>

[See the legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)

## You are free to

1. **Share** — copy and redistribute the material in any medium or format for any purpose, even commercially.
2. **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

The licensor cannot revoke these freedoms as long as you follow the license terms.

## Under the following terms

1. **Attribution** — You must give [appropriate credit](https://creativecommons.org/licenses/by/4.0/#ref-appropriate-credit) , provide a link to the license, and [indicate if changes were made](https://creativecommons.org/licenses/by/4.0/#ref-indicate-changes) . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
2. **No additional restrictions** — You may not apply legal terms or [technological measures](https://creativecommons.org/licenses/by/4.0/#ref-technological-measures) that legally restrict others from doing anything the license permits.

## Notices

You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable [exception or limitation](https://creativecommons.org/licenses/by/4.0/deed.en#ref-exception-or-limitation) .

No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as [publicity, privacy, or moral rights](https://creativecommons.org/licenses/by/4.0/deed.en#ref-publicity-privacy-or-moral-rights) may limit how you use the material.

This deed highlights only some of the key features and terms of the actual license. It is not a license and has no legal value. You should carefully review all of the terms and conditions of the actual license before using the licensed material.

:::

:::{admonition} MIT for source code and code snippets
:class: attention dropdown

MIT License

Copyright (c) 2026, Mimer AI Factory project, The contributors

Copyright (c) 2025, ENCCS, The contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

:::

::::
