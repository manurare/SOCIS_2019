# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 26/7/19
library(shiny)
library(shinydashboard)
library(leaflet)


df <- readRDS("./correlation.rds")

ui <- dashboardPage(
dashboardHeader(),
dashboardSidebar(),
dashboardBody(
tags$style(type = "text/css", "#map {height: calc(100vh - 80px) !important;}"),
leafletOutput("map")
)
)

# ui <- fluidPage(
#     leafletOutput("mymap",height = 200)
# )

server <- function(input, output) {
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

runApp(shinyApp(ui=ui, server=server), launch.browser = TRUE)


        # addMarkers(lng = df$long,
        # lat = df$lat, clusterOptions = markerClusterOptions(),
        # popup = paste("Weight", df$val, "<br>"))

