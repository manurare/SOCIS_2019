# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 6/8/19
library(shiny)
library(shinydashboard)
library(leaflet)

server <- function(input, output) {
    df <- readRDS("./correlation.rds")
    data <- reactive({
        x <- df
    })

    output$map <- renderLeaflet({
        df <- data()

        leaflet(data=df) %>%
        addTiles() %>%
            addCircleMarkers(radius = df$val, popup = paste("Weight", df$value, "<br>"))


    })
}
