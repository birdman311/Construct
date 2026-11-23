// SPDX-FileCopyrightText: 2026 birdman311
// SPDX-License-Identifier: MIT

#include "yuzu/glyph_ribbon.h"

#include <cmath>

#include <QAction>
#include <QApplication>
#include <QDate>
#include <QHelpEvent>
#include <QImage>
#include <QLinearGradient>
#include <QMenuBar>
#include <QMouseEvent>
#include <QPainter>
#include <QRandomGenerator>
#include <QTime>
#include <QToolTip>

namespace {
// Must match ATLAS in tools/theme/make_construct_script_art.py: 26 letters, '-', '*' (spiral curl).
constexpr char kAtlasOrder[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-*";
constexpr int kCell = 128;
constexpr double kSpeed = 26.0; // px per second
constexpr int kRepeats = 3;     // each message passes this many times before the next one
const QColor kMint{92, 232, 176};
const QColor kGold{232, 200, 108};

int AtlasIndex(QChar c) {
    const char ch = c.toUpper().toLatin1();
    for (int i = 0; kAtlasOrder[i]; ++i) {
        if (kAtlasOrder[i] == ch) {
            return i;
        }
    }
    return -1;
}

QStringList MessagesFor(GlyphRibbon::Place place) {
    QStringList list;
    if (place == GlyphRibbon::Place::MenuBar) {
        list = {QStringLiteral("THE CONSTRUCTS AWAKEN"), QStringLiteral("AGE OF IMPRISONMENT"),
                QStringLiteral("POWER OF THE ANCIENT ONES"),
                QStringLiteral("GUIDED BY THE LIGHT OF THE ZONAI")};
    } else {
        list = {QStringLiteral("READY"), QStringLiteral("CONSTRUCT SWITCH EMULATOR"),
                QStringLiteral("MAY THE STONE GUIDE YOUR HAND"),
                QStringLiteral("EVERY CONSTRUCT REMEMBERS ITS MAKER")};
    }
    // easter eggs that depend on the clock
    const QDate today = QDate::currentDate();
    if (today.month() == 11 && today.day() == 22) {
        list.prepend(QStringLiteral("HAPPY BIRTHDAY CONSTRUCT"));
    }
    if (QTime::currentTime().hour() < 4) {
        list.append(QStringLiteral("THE CONSTRUCTS NEVER SLEEP"));
    }
    return list;
}

// Rare lines, drawn in gold.
const QStringList kSecrets = {
    QStringLiteral("IF YOU CAN READ THIS YOU KNOW THE CODE"),
    QStringLiteral("THE SPIRAL HAS NO END"),
    QStringLiteral("A CONSTRUCT NEVER FORGETS"),
    QStringLiteral("LOOK CLOSER TRAVELER"),
    // the user's ten hidden messages (10 is not random, see below)
    QStringLiteral("I CANT BELIEVE YOU DECIPHERED THIS"),
    QStringLiteral("MORE UPDATES COMING SOON"),
    QStringLiteral("I HOPE YOU ENJOY PLAYING"),
    QStringLiteral("PLEASE REPORT OR FIX ANY BUGS YOU FIND"),
    QStringLiteral("RIP SONIA"),
    QStringLiteral("YOU WONT BELIEVE THE CUT CONTENT IN THE FILES"),
    QStringLiteral("PLEASE SUPPORT ME IN ANY WAY"),
    QStringLiteral("RIP KNIGHT CONSTRUCT"),
    QStringLiteral("MORE ON THE WAY"),
};
// Message 10: after 10 hours with the GUI open, or on returning to the GUI after a 5+ hour game.
const QString kTouchGrass = QStringLiteral("DUDE TOUCH SOME GRASS");
constexpr qint64 kHourMs = 60LL * 60 * 1000;
} // namespace

GlyphRibbon::GlyphRibbon(Place place_, QWidget* parent) : QWidget(parent), place{place_} {
    setAttribute(Qt::WA_TransparentForMouseEvents, false);
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::NoFocus);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    setMinimumWidth(0);
    atlas = QPixmap(QStringLiteral(":/construct_qss/rc/art_glyphs.png"));
    messages = MessagesFor(place);
    repeats_left = kRepeats;
    timer.setInterval(33);
    timer.setTimerType(Qt::CoarseTimer);
    connect(&timer, &QTimer::timeout, this, &GlyphRibbon::Tick);
    clock.start();
    next_grass_ms = 10 * kHourMs;

    menu_bar = place == Place::MenuBar ? qobject_cast<QMenuBar*>(parent) : nullptr;
    if (menu_bar) {
        menu_bar->installEventFilter(this);
        QTimer::singleShot(0, this, &GlyphRibbon::PlaceInMenuBar);
    }
    RefreshTheme();
}

void GlyphRibbon::RefreshTheme() {
    theme_active = !atlas.isNull() && qApp->styleSheet().contains(QStringLiteral("construct_qss"));
    setVisible(theme_active);
    if (theme_active) {
        Rebuild();
        if (menu_bar) {
            PlaceInMenuBar();
        }
    }
    UpdateTimer();
}

void GlyphRibbon::SetPaused(bool paused_) {
    if (paused_ && !paused) {
        game_started_ms = clock.elapsed();
    } else if (!paused_ && paused && game_started_ms >= 0) {
        if (clock.elapsed() - game_started_ms >= 5 * kHourMs) {
            QueueSecret(kTouchGrass);
        }
        game_started_ms = -1;
    }
    paused = paused_;
    UpdateTimer();
}

void GlyphRibbon::UpdateTimer() {
    const bool run = theme_active && !paused && isVisible() && width() > 0;
    if (run && !timer.isActive()) {
        last_ms = clock.elapsed();
        timer.start();
    } else if (!run && timer.isActive()) {
        timer.stop();
    }
}

void GlyphRibbon::PlaceInMenuBar() {
    if (!menu_bar) {
        return;
    }
    int right = 0;
    for (QAction* action : menu_bar->actions()) {
        if (action->isVisible()) {
            right = std::max(right, menu_bar->actionGeometry(action).right());
        }
    }
    const int x = right + 14;
    const int w = std::max(0, menu_bar->width() - x - 6);
    setGeometry(x, 0, w, menu_bar->height());
    raise();
}

bool GlyphRibbon::eventFilter(QObject* watched, QEvent* event) {
    if (watched == menu_bar &&
        (event->type() == QEvent::Resize || event->type() == QEvent::LayoutRequest ||
         event->type() == QEvent::Show || event->type() == QEvent::ActionAdded ||
         event->type() == QEvent::ActionChanged || event->type() == QEvent::FontChange)) {
        QTimer::singleShot(0, this, &GlyphRibbon::PlaceInMenuBar);
    }
    return QWidget::eventFilter(watched, event);
}

QString GlyphRibbon::NextMessage(bool& gold) {
    gold = false;
    if (!pending_secrets.isEmpty()) {
        gold = true;
        return pending_secrets.takeFirst();
    }
    if (repeats_left <= 0) {
        repeats_left = kRepeats;
        message_index = (message_index + 1) % messages.size();
        // one in fifteen message changes slips in a gold secret line
        if (QRandomGenerator::global()->bounded(15) == 0) {
            gold = true;
            return kSecrets[QRandomGenerator::global()->bounded(int(kSecrets.size()))];
        }
    }
    --repeats_left;
    return messages[message_index];
}

GlyphRibbon::Segment GlyphRibbon::MakeSegment(const QString& text, bool gold) const {
    const int h = std::max(8, height());
    const double glyph = h * 0.98;   // drawn cell size; the glyph fills ~62% of a cell
    const double step = h * 0.70;    // advance per character
    const QString unit = text + QStringLiteral("  *  ");
    const int w = int(std::ceil(unit.size() * step));
    const qreal dpr = devicePixelRatioF();

    QImage img(QSize(int(w * dpr), int(h * dpr)), QImage::Format_ARGB32_Premultiplied);
    img.setDevicePixelRatio(dpr);
    img.fill(Qt::transparent);
    {
        QPainter p(&img);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        for (int i = 0; i < unit.size(); ++i) {
            const double cx = (i + 0.5) * step;
            const int idx = AtlasIndex(unit[i]);
            if (idx >= 0) {
                const double cy = h / 2.0 + std::sin(i * 0.7) * h * 0.05;
                const QRectF dst(cx - glyph / 2, cy - glyph / 2, glyph, glyph);
                p.drawPixmap(dst, atlas, QRectF(idx * kCell, 0, kCell, kCell));
            }
        }
        // tint the white art
        p.setCompositionMode(QPainter::CompositionMode_SourceIn);
        p.fillRect(QRectF(0, 0, w, h), gold ? kGold : kMint);
    }
    return Segment{QPixmap::fromImage(img), text, w};
}

void GlyphRibbon::Rebuild() {
    segments.clear();
    head_x = 0.0;
    Fill();
    update();
}

void GlyphRibbon::Fill() {
    if (!theme_active || height() <= 0) {
        return;
    }
    double end = head_x;
    for (const auto& s : segments) {
        end += s.width;
    }
    while (end < width() + 1) {
        bool gold = false;
        const QString text = NextMessage(gold);
        segments.push_back(MakeSegment(text, gold));
        end += segments.back().width;
    }
}

void GlyphRibbon::QueueSecret(const QString& text) {
    if (pending_secrets.contains(text)) {
        return;
    }
    pending_secrets.append(text);
    // drop segments that have barely entered so the secret comes in soon
    double x = head_x;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (x > width() * 0.6 && i > 0) {
            segments.resize(i);
            break;
        }
        x += segments[i].width;
    }
    Fill();
}

void GlyphRibbon::CheckUptime() {
    if (clock.elapsed() >= next_grass_ms) {
        next_grass_ms += 10 * kHourMs;
        QueueSecret(kTouchGrass);
    }
}

void GlyphRibbon::Tick() {
    CheckUptime();
    if (window()->isMinimized()) {
        return;
    }
    const qint64 now = clock.elapsed();
    const double dt = std::min<qint64>(now - last_ms, 200) / 1000.0;
    last_ms = now;
    const double speed = now < boost_until_ms ? kSpeed * 7 : kSpeed;
    head_x -= speed * dt;
    while (!segments.empty() && head_x + segments.front().width < 0) {
        head_x += segments.front().width;
        segments.pop_front();
    }
    Fill();
    update();
}

void GlyphRibbon::paintEvent(QPaintEvent*) {
    if (!theme_active || segments.empty()) {
        return;
    }
    const qreal dpr = devicePixelRatioF();
    QImage frame(QSize(int(width() * dpr), int(height() * dpr)), QImage::Format_ARGB32_Premultiplied);
    frame.setDevicePixelRatio(dpr);
    frame.fill(Qt::transparent);
    {
        QPainter p(&frame);
        double x = head_x;
        for (const auto& s : segments) {
            // the belt runs left to right: belt position x is measured from the right edge
            p.drawPixmap(QPointF(width() - x - s.width, 0), s.pixmap);
            x += s.width;
        }
        // fade both ends so glyphs drift in and out
        const double fade = std::min(48.0, width() / 6.0);
        QLinearGradient g(0, 0, width(), 0);
        g.setColorAt(0.0, QColor(0, 0, 0, 0));
        g.setColorAt(fade / std::max(1, width()), QColor(0, 0, 0, 255));
        g.setColorAt(1.0 - fade / std::max(1, width()), QColor(0, 0, 0, 255));
        g.setColorAt(1.0, QColor(0, 0, 0, 0));
        p.setCompositionMode(QPainter::CompositionMode_DestinationIn);
        p.fillRect(rect(), g);
    }
    QPainter(this).drawImage(QPoint(0, 0), frame);
}

bool GlyphRibbon::event(QEvent* event) {
    // Hidden: hold Shift while hovering to read the glyphs in plain letters.
    if (event->type() == QEvent::ToolTip) {
        auto* help = static_cast<QHelpEvent*>(event);
        if (QApplication::keyboardModifiers() & Qt::ShiftModifier) {
            double x = head_x;
            for (const auto& s : segments) {
                const double left = width() - x - s.width;
                if (help->pos().x() >= left && help->pos().x() < left + s.width) {
                    QToolTip::showText(help->globalPos(), s.text, this);
                    return true;
                }
                x += s.width;
            }
        }
        QToolTip::hideText();
        event->ignore();
        return true;
    }
    return QWidget::event(event);
}

void GlyphRibbon::mousePressEvent(QMouseEvent* event) {
    // Hidden: click the ribbon seven times quickly to open the seventh seal.
    const qint64 now = clock.elapsed();
    clicks.push_back(now);
    while (!clicks.empty() && now - clicks.front() > 3000) {
        clicks.pop_front();
    }
    if (clicks.size() >= 7) {
        clicks.clear();
        QueueSecret(QStringLiteral("THE SEVENTH SEAL IS OPEN"));
    }
    QWidget::mousePressEvent(event);
}

void GlyphRibbon::mouseDoubleClickEvent(QMouseEvent* event) {
    // Hidden: double-click makes the belt rush for a moment.
    boost_until_ms = clock.elapsed() + 2500;
    QWidget::mouseDoubleClickEvent(event);
}

void GlyphRibbon::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (!segments.empty() && segments.front().pixmap.height() !=
                                 int(std::max(8, height()) * devicePixelRatioF())) {
        Rebuild();
    } else {
        Fill();
    }
    UpdateTimer();
}

void GlyphRibbon::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    UpdateTimer();
}

void GlyphRibbon::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
    UpdateTimer();
}
