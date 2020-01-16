from typing_extensions import Literal
import plotly.graph_objects as go
import pandas as pd
import datetime
import calendar

ChartKind = Literal["bar", "pie"]
ThreeAxisKind = Literal["bar", "line", "dot"]


def largest_chat(values, kind: ChartKind = "bar", n: int = 10):
    if n != None:
        n = len(values.index) if len(values.index) <= n else n
    to_show = values.iloc[0:n]
    if n != None:
        other_count = values.iloc[n:]["message_count"].sum()
        to_show = to_show.append(
            {"title": "Other chats", "message_count": other_count}, ignore_index=True)
    if kind == "bar":
        graph = go.Bar(x=to_show["title"], y=to_show["message_count"])
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Chat")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=to_show["title"], values=to_show["message_count"], title=f"{n} Largest Chats", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure

def calls_by_chat(values, kind: ChartKind = "bar", n: int = 10):
    if n != None:
        n = len(values.index) if len(values.index) <= n else n
    to_show = values.iloc[0:n]
    if n != None:
        other_count = values.iloc[n:]["call_count"].sum()
        to_show = to_show.append(
            {"title": "Other chats", "call_count": other_count}, ignore_index=True)
    if kind == "bar":
        graph = go.Bar(x=to_show["title"], y=to_show["call_count"])
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Chat")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Calls")))
    elif kind == "pie":
        graph = go.Pie(
            labels=to_show["title"], values=to_show["message_count"], title=f"{n} Largest Chats by Calls", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure

def call_duration_by_chat(values, kind: ChartKind = "bar", n: int = 10):
    n = len(values.index) if len(values.index) <= n else n
    to_show = values.iloc[0:n]
    if n != None:
        other_duration = values.iloc[n:]["duration"].sum()
        to_show = to_show.append(
            {"title": "Other chats", "duration": other_duration}, ignore_index=True)
    if kind == "bar":
        graph = go.Bar(x=to_show["title"], y=to_show["duration"])
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Chat")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Duration of Calls (seconds)")))
    elif kind == "pie":
        graph = go.Pie(
            labels=to_show["title"], values=to_show["duration"], title=f"{n} Largest Chats by Call Duration (seconds)", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure

def messages_by_sender(values, kind: ChartKind = "bar", n: int = 10):
    if n != None:
        n = len(values.index) if len(values.index) <= n else n
    to_show = values.iloc[0:n]
    if n != None:
        other_count = values.iloc[n:]["message_count"].sum()
        to_show = to_show.append(
            {"sender": "Other people", "message_count": other_count}, ignore_index=True)
    if kind == "bar":
        graph = go.Bar(x=to_show["sender"], y=to_show["message_count"])
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Sender"), type="category"),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=to_show["sender"], values=to_show["message_count"], title=f"Top {n} senders by message count", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure


def chats_by_participant(values, kind: ChartKind = "bar", n: int = 5):
    if n != None:
        n = len(values.index) if len(values.index) <= n else n
    to_show = values.iloc[0:n]
    if kind == "bar":
        graph = go.Bar(x=to_show.index, y=to_show)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Sender"), type="category"),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Chats")))
    elif kind == "pie":
        graph = go.Pie(
            labels=to_show.index, values=to_show, title=f"Top {n} senders by chat count", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure


def message_count_by_year(values, kind: ChartKind = "bar"):
    yearly = values.groupby("year").size()
    if kind == "bar":
        graph = go.Bar(x=yearly.index, y=yearly.values)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Year")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=yearly.index, values=yearly.values, title=f"Number of Messages by Year", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure


def message_count_by_month(values, kind: ChartKind = "bar", show_year: bool = True):
    values = values.sort_values(["month_num", "year"])
    if show_year:
        monthly = values.groupby(["year", "month"], sort=False).size()
    else:
        monthly = values.groupby(["month"], sort=False).size()
    if kind == "bar":
        graph = go.Bar(x=monthly.index.get_level_values(
            "month"), y=monthly.values)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Month"), categoryorder="array", categoryarray=list(calendar.month_abbr)),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=monthly.index, values=monthly.values, title=f"Number of Messages by Month", sort=True)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")
    return figure


def message_count_by_hour(values, kind: ChartKind = "bar"):
    hourly = values.groupby("hour").size()
    if kind == "bar":
        graph = go.Bar(x=hourly.index, y=hourly.values)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Hour")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=hourly.index, values=hourly.values, title=f"Number of Messages by Hour", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")

    return figure


def message_count_by_weekday(values, kind: ChartKind = "bar"):
    by_weekday = values.groupby(["day_of_week_num", "day_of_week"]).size()
    if kind == "bar":
        graph = go.Bar(x=by_weekday.index.get_level_values(
            "day_of_week"), y=by_weekday.values)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Day"), categoryorder="array", categoryarray=list(calendar.day_abbr)[1:]),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Messages")))
    elif kind == "pie":
        graph = go.Pie(
            labels=by_weekday.index.get_level_values("day_of_week"), values=by_weekday.values, title=f"Number of Messages by Weekday", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")
    return figure


def messages_by_day_by_chat(values, kind: ThreeAxisKind = "dot"):
    figure = go.Figure()
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Scatter(x=count.index.get_level_values(
                "date"), y=count, name=idx, mode=mode))

    elif kind == "bar":
        for idx, count in values.groupby(level="title"):
            figure.add_trace(
                go.Bar(x=count.index.get_level_values("date"), y=count, name=idx))
        figure.update_layout(barmode="stack", bargap=0)
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure


def messages_by_hour_by_chat(values, kind: ThreeAxisKind = "dot"):
    figure = go.Figure()
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Scatter(x=count.index.get_level_values(
                "hour"), y=count, name=idx, mode=mode))
    elif kind == "bar":
        for idx, count in values.groupby(level="title"):
            figure.add_trace(
                go.Bar(x=count.index.get_level_values("hour"), y=count, name=idx))
        figure.update_layout(barmode="stack")
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure


def messages_by_weekday_by_chat(values, kind: ThreeAxisKind = "bar"):
    figure = go.Figure()
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Scatter(x=count.index.get_level_values(
                "weekday"), y=count, name=idx, mode=mode))
    elif kind == "bar":
        for idx, count in values.groupby(level="title"):
            figure.add_trace(
                go.Bar(x=count.index.get_level_values("weekday"), y=count, name=idx))
        figure.update_layout(barmode="stack")
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure


def messages_by_hour_of_week_by_chat(values, kind: ThreeAxisKind = "bar"):
    figure = go.Figure()
    axis = [[calendar.day_abbr[j]
             for j in range(0, 7) for i in range(0, 24)], list(range(0, 24)) * 7]
    AXIS_ORDER = [(day, hour) for day in calendar.day_abbr for hour in range(0,24)]
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        figure.add_trace(go.Scatter(
            x=axis, y=[0 for i in range(0, 24*7)], name="", mode=mode, marker_color="#000000"))
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Scatter(x=[count.index.get_level_values(
                "day"), count.index.get_level_values("hour")], y=count, name=idx, mode=mode))
    elif kind == "bar":
        figure.add_trace(
            go.Bar(x=axis, y=[0 for i in range(0, 24*7)], name=""))
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Bar(x=[count.index.get_level_values(
                "day"), count.index.get_level_values("hour")], y=count, name=idx))
        figure.update_layout(barmode="stack")
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    figure.update_layout(xaxis=dict(categoryorder="array", categoryarray=AXIS_ORDER))
    return figure

def messages_by_chat_hourly(values, kind: ThreeAxisKind = "line"):
    figure = go.Figure()
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        for idx, count in values.groupby(level="title"):
            figure.add_trace(go.Scatter(x=count.index.get_level_values(
                "hourly"), y=count, name=idx, mode=mode))
    elif kind == "bar":
        for idx, count in values.groupby(level="title"):
            figure.add_trace(
                go.Bar(x=count.index.get_level_values("hourly"), y=count, name=idx))
        figure.update_layout(barmode="stack")
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure

def reaction_counts(values, kind: ChartKind = "bar"):
    if kind == "bar":
        graph = go.Bar(x=values.index, y=values.values)
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Reaction")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Number of Reactions")))

    elif kind == "pie":
        graph = go.Pie(
            labels=values.index, values=values.values, title=f"Reactions", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")
    return figure


def reactions_by_day(values, kind: ThreeAxisKind = "dot"):
    figure = go.Figure()
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
        for idx, count in values.groupby(level="reaction"):
            figure.add_trace(go.Scatter(x=count.index.get_level_values(
                "date"), y=count, name=idx, mode=mode))
    elif kind == "bar":
        for idx, count in values.groupby(level="reaction"):
            figure.add_trace(
                go.Bar(x=count.index.get_level_values("date"), y=count, name=idx))
        figure.update_layout(barmode="stack")
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure


def most_frequent_words(values, kind: ChartKind = "bar", min_length: int = 5, n: int = 5):
    filtered = values[[len(row["word"]) >= min_length for _,
                       row in values.iterrows()]]
    if n != None:
        n = len(filtered.index) if len(filtered.index) <= n else n
    filtered = filtered[0:n]
    if kind == "bar":
        graph = go.Bar(x=filtered["word"], y=filtered["count"])
        figure = go.Figure(graph)
        figure.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Word")),
                             yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Frequency")))

    elif kind == "pie":
        graph = go.Pie(
            labels=values.index, values=values.values, title=f"Words with at least {min_length} letters", sort=False)
        figure = go.Figure(graph)
    else:
        raise ValueError("kind must be either 'bar' or 'pie'")
    return figure

def most_frequent_words_by_day(values, kind: ThreeAxisKind = "line", min_length: int = 5, n: int = 5):
    filtered = values[[len(row["word"]) >= min_length for _,
                       row in values.iterrows()]]
    if n != None:
        n = len(filtered.index) if len(filtered.index) <= n else n
    filtered = filtered[0:n]
    if kind == "dot" or kind == "line":
        mode = "markers" if kind == "dot" else "lines+markers"
    elif kind == "bar":
        return 0
    else:
        raise ValueError("kind must be either 'dot', 'line' or 'bar'")
    return figure

def top_chats_per_day(values):
    figure = go.Figure()
    for idx, rank in values.groupby(level="title"):
        figure.add_trace(go.Scatter(x=rank.index.get_level_values(
                "date"), y=rank,mode="lines+markers", name=idx, connectgaps=False))
    figure.update_layout(yaxis=dict(autorange = "reversed"))
    return figure
