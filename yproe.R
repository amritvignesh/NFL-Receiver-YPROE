library(nflreadr)
library(dplyr)
library(xgboost)
library(caret)
library(vip)
library(nflplotR)
library(gt)
library(gtExtras)

receiving_stats <- load_nextgen_stats(stat_type = "receiving") 

receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Cooper Kupp")] <- "LAR"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Davante Adams")] <- "LV"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Tyreek Hill")] <- "KC"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Justin Jefferson")] <- "MIN"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Mark Andrews")] <- "BAL"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Diontae Johnson")] <- "PIT"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Keenan Allen")] <- "LAC"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Stefon Diggs")] <- "BUF"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Travis Kelce")] <- "KC"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Ja'Marr Chase")] <- "CIN"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Deebo Samuel")] <- "SF"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Mike Evans")] <- "TB"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "George Kittle")] <- "SF"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Kyle Pitts")] <- "ATL"
receiving_stats$team_abbr[which(receiving_stats$season == 2021 & receiving_stats$player_display_name == "Cooper Kupp")] <- "LAR"

receiving_stats_initial <- receiving_stats %>%
  filter(week != 0) %>%
  group_by(season, player_gsis_id) %>%
  filter(n_distinct(team_abbr) != 1) %>%
  distinct(player_gsis_id)

receiving_stats <- receiving_stats %>%
  filter(week == 0) %>%
  anti_join(receiving_stats_initial, by = c("season", "player_gsis_id")) %>%
  mutate(ypr = yards/receptions) %>%
  group_by(season) %>%
  arrange(-receptions) %>%
  filter(row_number() <= 100) %>%
  ungroup() %>%
  select(season, player_gsis_id, player_display_name, player_position, team_abbr, avg_cushion, avg_separation, avg_intended_air_yards, avg_expected_yac, receptions, ypr)

passing_stats <- load_nextgen_stats(stat_type = "passing") 

passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Aaron Rodgers")] <- "GB"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Kirk Cousins")] <- "MIN"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Russell Wilson")] <- "SEA"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Patrick Mahomes")] <- "KC"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Justin Herbert")] <- "LAC"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Kyler Murray")] <- "ARI"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Lamar Jackson")] <- "BAL"
passing_stats$team_abbr[which(passing_stats$season == 2021 & passing_stats$player_display_name == "Tom Brady")] <- "TB"

passing_stats_initial <- passing_stats %>%
  filter(week != 0) %>%
  group_by(season, player_gsis_id) %>%
  filter(n_distinct(team_abbr) != 1) %>%
  distinct(player_gsis_id)

passing_stats <- passing_stats %>%
  filter(week == 0) %>%
  anti_join(passing_stats_initial, by = c("season", "player_gsis_id")) %>%
  mutate(total_aggressiveness = aggressiveness * attempts, total_expected_completion_percentage = expected_completion_percentage * attempts) %>%
  group_by(season, team_abbr) %>%
  summarize(aggressiveness = sum(total_aggressiveness)/sum(attempts), expected_completion_percentage = total_expected_completion_percentage/attempts) %>%
  distinct(team_abbr, .keep_all = TRUE)

receiving_stats <- left_join(receiving_stats, passing_stats, by = c("season", "team_abbr"))

receiving_stats$player_position <- as.factor(receiving_stats$player_position)

factor_data <- receiving_stats %>% select(position = player_position)

dummy <- dummyVars(" ~ .", data = factor_data)
factor_data <- data.frame(predict(dummy, newdata = factor_data))

receiving_stats <- cbind(receiving_stats, factor_data) 

xgboost_train <- receiving_stats %>%
  filter(season < 2022)

xgboost_test <- receiving_stats %>%
  filter(season >= 2022)

labels_train <- as.matrix(xgboost_train[, 11])
xgboost_trainfinal <- as.matrix(xgboost_train[, c(6:9, 12:15)])
xgboost_testfinal <- as.matrix(xgboost_test[, c(6:9, 12:15)])

yoe_model <- xgboost(data = xgboost_trainfinal, label = labels_train, nrounds = 100, objective = "reg:squarederror", early_stopping_rounds = 10, max_depth = 6, eta = 0.3)

vip(yoe_model)

ypr_predict <- predict(yoe_model, xgboost_testfinal)
ypr <- as.matrix(xgboost_test[,11])
postResample(ypr_predict, ypr)

ypr_predictions <- as.data.frame(
  matrix(predict(yoe_model, as.matrix(receiving_stats[,c(6:9, 12:15)])))
)

all_stats <- cbind(receiving_stats, ypr_predictions) %>%
  rename(predicted_ypr = V1) %>%
  mutate(yproe = ypr - predicted_ypr) %>%
  select(season, id = player_gsis_id, name = player_display_name, team = team_abbr, position = player_position, receptions, ypr, xypr = predicted_ypr, yproe)

stats_2023 <- all_stats %>%
  filter(season == 2023) %>%
  group_by(team) %>%
  filter(row_number() == 1) %>%
  select(-season)

gt_nice_stats <- stats_2023 %>%
  arrange(-yproe) %>%
  mutate(ypr = round(ypr, 1), xypr = round(xypr, 1), yproe = round(yproe, 2)) 

gt_align_caption <- function(left, right) {
  caption <- paste0(
    '<span style="float: left;">', left, '</span>',
    '<span style="float: right;">', right, '</span>'
  )
  return(caption)
}

caption = gt_align_caption("Data from <b>nflverse (NextGenStats)</b>", "Amrit Vignesh")

nice_table <- gt_nice_stats %>% gt() %>%
  gt_nfl_logos(id, height = 40) %>%
  gt_nfl_headshots(columns = id, height = 50) %>%
  gt_theme_538() %>%
  cols_align(
    align = "center",
    columns = c(id, name, team, position, receptions, ypr, xypr, yproe)
  ) %>%
  gt_hulk_col_numeric(c(receptions, ypr, xypr, yproe)) %>%
  cols_label(
    id = md(""),
    name = md("**Receiver**"),
    team = md("**Team**"),
    position = md("**Position**"),
    receptions = md("**Receptions**"),
    ypr = md("**YPR**"),
    xypr = md("**xYPR**"),
    yproe = md("**YPROE**")
  ) %>%
  tab_header(
    title = "2023 NFL Receiver YPROE (Yards Per Reception Over Expected)",
    subtitle = md("*Receiver with Most **Receptions** Per Team Displayed, **Regular Season** Only*")
  ) %>% 
  tab_source_note(html(caption)) %>%
  opt_align_table_header(align = "center") %>%
  tab_style(
    style = list(
      cell_text(weight = "bold")
    ),
    locations = cells_body(
      columns = c(name, position, yproe)
    )
  ) 

gtsave(nice_table, "nice_table.png", vwidth = 1000, vheight = 2500, zoom = 1)
