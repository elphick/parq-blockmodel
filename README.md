# elphick-template

This is a template for a python package including:

- a namespace directory (called namespace)
- a package directory (called package)
- tests
- docs
- github actions

Actions to perform once you have created your repo from this template:

1. Change references to elphick-template, including in the pyproject.toml file
2. Change the namespace and package folder names and module.py filename
3. Modify the content of the following rst pages:
    - api/modules
    - installation
    - quickstart
    - glossary
4. Confirm the licence file and modify accordingly
5. Consider moving matplotlib and plotly from dev dependencies to your package dependencies if you use them.

One of the advantages of the template is the doc publishing onto a gh-pages branch is already configured.
To leverage this be sure to check the "include all branches" checkbox when creating a new repository from the template.

[![screenshot](https://elphick.github.io/elphick-template/_static/new_repo_from_template.png)](https://elphick.github.io/elphick-template/_static/new_repo_from_template.png)

Oh, you'll likely need to set-up a github token for the docs_to_gh_pages.yml action to work.
TODO: Confirm and add better instruction here.
