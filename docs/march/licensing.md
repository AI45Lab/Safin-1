# Code release licensing

The Safin-1 code release retains the ms-swift fork's **Apache-2.0** license.
The root LICENSE, NOTICE, package metadata and README use that same code license.
This is the selected code-release policy. The original landing page's MIT notice
is retained separately for the original materials it covers.

Each vendored project keeps its own LICENSE/LICENSE.md, NOTICE and source/change
records. The [component index](../../third_party/VENDORED_COMPONENTS.md) and
[license inventory](source-audit-20261017.json) cover all six components, including
the optional historical research dependencies.

The public landing repository contained an MIT license. Its exact original
notice is retained in
[LICENSES/AI45Lab-Safin-1-MIT.txt](../../LICENSES/AI45Lab-Safin-1-MIT.txt).
When merging this code into the public repository, preserve the existing landing
materials and keep that original notice with them. Use the delivered Apache-2.0
LICENSE and NOTICE for the new code distribution. The source archive, sdist and
project wheel must preserve the applicable license files. The original landing history and technical PDF are retained in the repository.

Weights, baselines and training data have separate licenses on their model/data
pages. This code-release decision does not assign those licenses.

The [release record](license-release-20261017.json) contains attribution-file
checksums and the exact source of the retained original MIT notice.
