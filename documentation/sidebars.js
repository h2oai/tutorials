module.exports = {
  defaultSidebar: [
    "index",
    {
      "Get started": [
        "get-started/access-aquarium",
      ],
    },
    {
      type: "category",
      label: "Tutorials",
      link: { type: "doc", id: "tutorials/tutorials-overview" },
      items: [
        {
          type: "category",
          label: "H2O Driverless AI core",
          items: [
            "tutorials/h2o-driverless-ai/tutorial-1a",
            "tutorials/h2o-driverless-ai/tutorial-2a",
            "tutorials/h2o-driverless-ai/tutorial-3a",
          ],
        },
        {
          type: "category",
          label: "Time series, NLP, and image processing",
          items: [
            "tutorials/time-series-nlp-and-image-processing/tutorial-1b",
            "tutorials/time-series-nlp-and-image-processing/tutorial-2b",
            "tutorials/time-series-nlp-and-image-processing/tutorial-3b",
          ],
        },
        {
          type: "category",
          label: "Recipes",
          items: [
            "tutorials/recipes/tutorial-1c",
            "tutorials/recipes/tutorial-2c",
            "tutorials/recipes/tutorial-3c",
          ],
        },
        {
          type: "category",
          label: "Model deployment",
          items: [
            "tutorials/model-deployment/tutorial-1d",
            "tutorials/model-deployment/tutorial-2d",
            "tutorials/model-deployment/tutorial-3d",
          ],
        },
        {
          type: "category",
          label: "Machine learning interpretability",
          items: [
            "tutorials/machine-learning-interpretability/tutorial-1e",
            "tutorials/machine-learning-interpretability/tutorial-2e",
            "tutorials/machine-learning-interpretability/tutorial-3e",
          ],
        },
      ],
    },
    "concepts",
    "key-terms",
    "faqs",

  ],
};

