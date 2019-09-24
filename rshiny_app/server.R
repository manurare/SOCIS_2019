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

server <- function(input, output) {

    test <<- NULL

    output$map <- renderLeaflet({
            leaflet() %>%
            addTiles()
        })


        observe({
            if(is.null(input$file)){return()}
                color_vec = c("red", "blue", "green", "yellow", "pink", "purple", "cyan")
                inputFile <- input$file$datapath
                df <- lapply(inputFile, readRDS)
                slider_vecNames <- input$file$name
                slider_valueNames <- paste(input$file$name, "Value", sep="")
                leafletProxy("map") %>%
                    clearControls() %>%
                    addLegend("bottomright", colors = color_vec[1:length(df)], labels = slider_vecNames,
                    title = "Layer",
                    labFormat = labelFormat(prefix = "$"),
                    opacity = 1)%>%
                    clearMarkers()
                    lapply(1:length(df), function(i){
                        current_slider_vis <- slider_vecNames[[i]]
                        current_slider_val <- slider_valueNames[[i]]
                        min_slider_val <- input[[current_slider_val]][[1]]
                        max_slider_val <- input[[current_slider_val]][[2]]
                        df_filtered <- df[[i]][df[[i]]$value >= min_slider_val & df[[i]]$value <= max_slider_val,]
                        if (length(input[[current_slider_vis]]) > 0){
                            if(input[[current_slider_vis]] != 0){
                        leafletProxy("map") %>%
                            addCircleMarkers(radius = df_filtered$val*input[[current_slider_vis]],
                                            lat = df_filtered$lat, lng = df_filtered$long,
                                            color=color_vec[[i]], stroke = FALSE, popup = paste("Weight", df[[i]]$value, "<br>"))
                            }
                        }
                    })
            })

    ###Print the values of the created sliders
    output$test <- renderText({
        if (is.null(input$file)) {return()}
        paste(lapply(1:length(input$file$name), function(i)
        {
            inputName <- input$file$name[i]
            input[[inputName]]
        }))
  })

    ###Create sliders dependent on the number of uploaded files
    output$sliders <- renderUI({
        if (is.null(input$file)) {return()}
        # First, create a list of sliders each with a different name
        lapply(1:length(input$file$name), function(i)
        {
            inputFile <- input$file$datapath[i]
            df <- readRDS(inputFile)
            max_slider <- max(unlist(df$value))
            min_slider <- min(unlist(df$value))
            inputName <- input$file$name[i]
            name_slider_vis <- strsplit(inputName, "[.]")[[1]][[1]]
            id_slider_val <- paste(inputName,"Value",sep="")
            lapply(1:2,function(j){
                if(j==1){
                    sliderInput(inputId = inputName, label = toupper(name_slider_vis), min=0, max=10, value=1)
                }else{
                    sliderInput(inputId = id_slider_val, label = paste("Filter Value",toupper(name_slider_vis)),
                            min=floor(min_slider), max=ceiling(max_slider), value=c(min,max))
                }
            })
        })
        # Create a tagList of sliders (this is important)
        # test <<- do.call(tagList, sliders)
    })
}