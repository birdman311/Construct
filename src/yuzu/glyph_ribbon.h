// SPDX-FileCopyrightText: 2026 birdman311
// SPDX-License-Identifier: MIT

#pragma once

#include <deque>

#include <QElapsedTimer>
#include <QPixmap>
#include <QStringList>
#include <QTimer>
#include <QWidget>

class QMenuBar;

// A thin conveyor belt of Construct's glyph script (original cipher alphabet, one glyph per Latin
// letter, art from tools/theme/make_construct_script_art.py). Messages scroll left-to-right and
// repeat. It only shows with the Construct theme and stops animating while a game runs.
class GlyphRibbon : public QWidget {
public:
    enum class Place { MenuBar, StatusBar };

    explicit GlyphRibbon(Place place, QWidget* parent);

    // Stops the animation (e.g. while a game runs). The ribbon still shows its last frame.
    void SetPaused(bool paused);
    // Re-check whether the Construct theme is active and show or hide the ribbon.
    void RefreshTheme();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    bool event(QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    struct Segment {
        QPixmap pixmap;
        QString text;
        int width = 0;
    };

    void Tick();
    void Fill();
    void Rebuild();
    void UpdateTimer();
    void PlaceInMenuBar();
    QString NextMessage(bool& gold);
    void QueueSecret(const QString& text);
    void CheckUptime();
    Segment MakeSegment(const QString& text, bool gold) const;

    Place place;
    QMenuBar* menu_bar = nullptr;
    QPixmap atlas;
    bool theme_active = false;
    bool paused = false;

    QStringList messages;
    int message_index = 0;
    int repeats_left = 0;
    QStringList pending_secrets;     // gold lines waiting to roll in next
    qint64 game_started_ms = -1;     // when the running game started (ribbon clock)
    qint64 next_grass_ms = 0;        // GUI uptime at which the next 10-hour line is due

    std::deque<Segment> segments;
    double head_x = 0.0;
    QTimer timer;
    QElapsedTimer clock;
    qint64 last_ms = 0;
    qint64 boost_until_ms = 0;
    std::deque<qint64> clicks;
};
