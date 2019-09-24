# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 6/8/19
library(shiny)
# library(reactlog)
# options(shiny.reactlog=TRUE)
library(shinydashboard)
library(leaflet)
library(raster)


ui <- dashboardPage(
dashboardHeader(),
dashboardSidebar(
    # Input: Select a file ----
    fileInput("file", "Choose RDS File",
    multiple = TRUE,
    accept = c(".rds")),
    # accept = c("text/csv",
    # "text/comma-separated-values,text/plain",
    # ".csv", ".rds")),

    # Horizontal line ----
    tags$hr(),
    # sliderInput("longitude", "Longitude", min=-180, max=180, value=c(-180, 180)),
    # sliderInput("latitude", "Latitude", min=-90, max=90, value=c(-90, 90)),
    # tags$hr(),


    # Input: Checkbox if file has header ----
    # checkboxInput("header", "Header", TRUE),
    # uiOutput("selectfile"),
    uiOutput("sliders"),
    textOutput("selected_var"),
    textOutput("test")
),
dashboardBody(
tags$style(type = "text/css", "#map {height: calc(100vh - 80px) !important;}"),
leafletOutput("map")
)
)
