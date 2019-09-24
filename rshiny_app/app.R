# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 26/7/19
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

# ui <- fluidPage(
#     leafletOutput("mymap",height = 200)
# )

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

runApp(shinyApp(ui=ui, server=server), launch.browser = TRUE)






######WORKS
    # df <- readRDS("./correlation.rds")
    # test <<- NULL
    #
    # data <- reactive({
    #     x <- df
    # })
    #
    # output$map <- renderLeaflet({
    #     df <- data()
    #
    #     leaflet(data=df) %>%
    #         addTiles() %>%
    #         addCircleMarkers(radius = df$val, color="red", stroke = FALSE,popup = paste("Weight", df$value, "<br>"))
    #
    #
    # })



    # observe({
    #     if(is.null(input$file)){return()}
    #     inputFile <- input$file$datapath
    #     df <- lapply(inputFile, readRDS)
    #     slider_vecNames <- input$file$name
    #     if (length(df) == 1){
    #         slider_name = slider_vecNames[[1]]
    #         dataset = df[[1]]
    #         # print(dataset$lat)
    #         if(!is.null(input[[slider_name]]))
    #         {
    #             if(input[[slider_name]] == 0)
    #             {
    #                 leafletProxy("map") %>%
    #                 clearMarkers()
    #             }else{
    #                 leafletProxy("map") %>%
    #                 clearMarkers() %>%
    #                     addCircleMarkers(radius = dataset$val*input[[slider_name]], lat = dataset$lat, lng = dataset$long,
    #                         color="red", stroke = FALSE,popup = paste("Weight", dataset$value, "<br>"))
    #                 }
    #         }
    #     }
    #     if (length(df) == 2){
    #         slider_name_1 = slider_vecNames[[1]]
    #         slider_name_2 = slider_vecNames[[2]]
    #         dataset_1 = df[[1]]
    #         dataset_2 = df[[2]]
    #         if(!is.null(input[[slider_name_1]]) && !is.null(input[[slider_name_2]])){
    #         leafletProxy("map") %>%
    #             clearMarkers() %>%
    #                 addCircleMarkers(radius = dataset_1$val*input[[slider_name_1]], lat = dataset_1$lat, lng = dataset_1$long,
    #                     color="red", stroke = FALSE,popup = paste("Weight", dataset_1$value, "<br>")) %>%
    #                 addCircleMarkers(radius = dataset_2$val*input[[slider_name_2]], lat = dataset_2$lat, lng = dataset_2$long,
    #                     color="blue", stroke = FALSE,popup = paste("Weight", dataset_2$value, "<br>"))
    #         }
    #     }
    #     if (length(df) == 3){
    #         slider_name_1 = slider_vecNames[[1]]
    #         slider_name_2 = slider_vecNames[[2]]
    #         slider_name_3 = slider_vecNames[[3]]
    #         dataset_1 = df[[1]]
    #         dataset_2 = df[[2]]
    #         dataset_3 = df[[3]]
    #         if(!is.null(input[[slider_name_1]]) && !is.null(input[[slider_name_2]]) && !is.null(input[[slider_name_3]])){
    #         leafletProxy("map") %>%
    #             clearMarkers() %>%
    #                 addCircleMarkers(radius = dataset_1$val*input[[slider_name_1]], lat = dataset_1$lat, lng = dataset_1$long,
    #                     color="red", stroke = FALSE,popup = paste("Weight", dataset_1$value, "<br>")) %>%
    #                 addCircleMarkers(radius = dataset_2$val*input[[slider_name_2]], lat = dataset_2$lat, lng = dataset_2$long,
    #                     color="blue", stroke = FALSE,popup = paste("Weight", dataset_2$value, "<br>")) %>%
    #                 addCircleMarkers(radius = dataset_3$val*input[[slider_name_3]], lat = dataset_3$lat, lng = dataset_3$long,
    #                     color="yellow", stroke = FALSE,popup = paste("Weight", dataset_3$value, "<br>"))
    #         }
    #     }
    # })