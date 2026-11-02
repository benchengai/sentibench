# Licences for the files in this folder

The cleaned splits here are derived from third-party datasets. They are not
covered by this repository's MIT licence for code; each keeps its source
licence. Changes made: whitespace stripped (and doubled quotes unescaped in
the GitHub set), one empty Jira row dropped, the GitHub set split 70/30 with a
fixed seed, and a dev set held out of each train split (see
`sentibench/data.py`).

## `github/`

GitHub gold standard, Novielli, N., Calefato, F., Dongiovanni, D., Girardi, D.,
Lanubile, F. (2020). Can We Use SE-specific Sentiment Analysis Tools in a
Cross-Platform Setting? MSR 2020. Data: figshare,
https://doi.org/10.6084/m9.figshare.11604597. Licence: CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/.

## `jira/` and `so/`

From collab-uniba/EMTK_datasets (commit 2592acc), which redistributes the Jira
dataset of Ortu et al. (2016) and the Stack Overflow polarity dataset of
Calefato et al. (2018, Senti4SD). Please cite those papers. The repository's
licence notice follows.

```
MIT License

Copyright (c) 2019 Collaborative Development Group

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
```
