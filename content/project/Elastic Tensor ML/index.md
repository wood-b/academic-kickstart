+++
title = "Elastic Tensor ML"
date = 2018-12-19T11:28:58-08:00
draft = false


# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["elastic tensor", "machine learning", "machine-learning"]

# Project summary to display on homepage.
summary = "Machine learning models for predicting the bulk modulus of materials"

# Slides (optional).
#   Associate this page with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references 
#   `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides = ""

# Optional external URL for project (replaces project detail page).
external_link = ""

# Links (optional).
url_pdf = ""
url_code = "https://github.com/wood-b/elastic_tensor_ML"
url_dataset = ""
url_slides = ""
url_video = ""
url_poster = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{icon_pack = "fab", icon="twitter", name="Follow", url = "https://twitter.com"}]

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
[image]
  # Caption (optional)
  caption = ""

  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = ""
+++

The Materials Project contains ~87,000 total materials with ~13,500 elastic tensors, which are computationally intensive to calculate from first principles. Although the total number computed will continue to grow, it would be nice to use the data we already have to predict elastic properties such as bulk and shear modulus for materials where the elastic tensor is yet to be calculated. As a result, the goals of this project are to compare machine learning (ML) models for predicting the bulk modulus, and to update the previous model that was trained on a smaller data set. For example Jupyter Notebooks follow the code link above.
