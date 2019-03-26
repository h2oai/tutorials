### Converting tutorials

This doc walks you through how to convert a properly-formatted tutorial (in markdown) to a codelab and subsequently add that generated codelab to the tutorials landing page.

## Required tutorial metadata

Minimum metadata that must be defined on tutorials
```

---
id: test-drive
feedback: https://github.com/h2oai/tutorials/issues
---
```

Expanded (optional) metadata
```

---
id: test-drive-codelabs0
summary: Check the list
categories: driverless
tags: driverless, test-drive, aquarium, security
duration: 40 (in minutes)
feedback: https://github.com/h2oai/tutorials/issues
status: draft/published/hidden
published: 2018-03-23
ga: GA-12345 (H2O's google analytics accnt)
authors: Edgar Orendain <edgar.orendain@h2o.ai>, Ana Castro <ana.castro@h2o.ai>
---
```


## Download the claat tool
1. Download the latest claat binary from from https://github.com/googlecodelabs/tools/releases (v2.0.2 at the time of this writing)
2. Run the following command to convert a markdown (.md) file into a codelab.
```
claat -f html -prefix "../assets" <relative-path-to-markdown-file> -ga <global GA account>
```

> Note: For `<global GA account>` (e.g. GA-12345), check with Marketing (not available at time of this writing)

This will generate a folder (titled after the generated tutorial) with all required files.  Drop this folder into the `gh-pages` branch of the tutorials repo.


## Add the tutorial to the tutorial landing page

1. Open `index.html` sitting on the root of the `gh-pages` repo.
2. Copy and paste a copy of one of the `<codelabs-card>` blocks. In the copy you created, look for and replace necessary text (title, summary, etc).  Specifically, be sure to change the `href` target to the name of the folder generated in the previous section.
3. Save this edited file.  Once pushed to github, changes should be reflected on `h2oai.github.io/tutorials`.


Questions? Contact Edgar Orendain <edgar.orendain@h2o.ai> or Ana Castro (ana.castro@h2o.ai)
