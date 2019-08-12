# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 6/8/19
library(shiny)
library(shinydashboard)
library(leaflet)

ui <- dashboardPage(
dashboardHeader(),
dashboardSidebar(),
dashboardBody(
tags$style(type = "text/css", "#map {height: calc(100vh - 80px) !important;}"),
leafletOutput("map")
)
)
