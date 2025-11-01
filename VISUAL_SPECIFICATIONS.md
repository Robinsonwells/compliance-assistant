# Chat Interface Visual Specifications

## Layout Dimensions

### Mobile (320px - 767px)
```
┌─────────────────────────────────┐
│ Header (56px)                   │
│ ┌──────────────┬──────┐         │
│ │ Title        │ [⬈] │         │
│ │ Subtitle     │      │         │
│ └──────────────┴──────┘         │
├─────────────────────────────────┤
│                                 │
│ Messages Container (Dynamic)    │
│                                 │
│ ┌─────────────────────────┐    │
│ │ Assistant Message       │    │
│ │ (90% width, left align) │    │
│ └─────────────────────────┘    │
│                                 │
│    ┌─────────────────────┐     │
│    │ User Message        │     │
│    │ (85% width, right)  │     │
│    └─────────────────────┘     │
│                                 │
├─────────────────────────────────┤
│ Fixed Input Bar (72px)          │
│ ┌────────────────────┬────┐    │
│ │ [Input Field]      │ [⬆]│    │
│ │ 56px height        │48px│    │
│ └────────────────────┴────┘    │
└─────────────────────────────────┘
```

### Tablet (768px - 1023px)
```
┌───────────────────────────────────────┐
│ Header (64px)                         │
│ ┌────────────────────┬────────┐       │
│ │ Title + Subtitle   │  [⬈]  │       │
│ └────────────────────┴────────┘       │
├───────────────────────────────────────┤
│                                       │
│   Messages Container (Centered)       │
│                                       │
│  ┌──────────────────────────────┐    │
│  │ Assistant Message            │    │
│  │ (80% width, left align)      │    │
│  └──────────────────────────────┘    │
│                                       │
│       ┌──────────────────────┐       │
│       │ User Message         │       │
│       │ (75% width, right)   │       │
│       └──────────────────────┘       │
│                                       │
├───────────────────────────────────────┤
│ Fixed Input Bar (Centered, 680px)    │
│    ┌──────────────────────┬────┐     │
│    │ [Input Field]        │ [⬆]│     │
│    └──────────────────────┴────┘     │
└───────────────────────────────────────┘
```

### Desktop (1024px+)
```
┌─────────────────────────────────────────────┐
│ Header (72px)                               │
│ ┌───────────────────────────┬──────────┐    │
│ │ Title + Subtitle          │   [⬈]   │    │
│ └───────────────────────────┴──────────┘    │
├─────────────────────────────────────────────┤
│                                             │
│     Messages Container (Max 1200px)         │
│                                             │
│   ┌─────────────────────────────────┐      │
│   │ Assistant Message               │      │
│   │ (75% width, left align)         │      │
│   └─────────────────────────────────┘      │
│                                             │
│            ┌────────────────────┐           │
│            │ User Message       │           │
│            │ (70% width, right) │           │
│            └────────────────────┘           │
│                                             │
├─────────────────────────────────────────────┤
│ Fixed Input Bar (Centered, 800px)          │
│      ┌──────────────────────┬────┐         │
│      │ [Input Field]        │ [⬆]│         │
│      └──────────────────────┴────┘         │
└─────────────────────────────────────────────┘
```

---

## Component Specifications

### Header Component
**Mobile:**
- Height: 56px
- Padding: 12px 16px
- Title: 1.75rem (28px)
- Logout: 44x44px icon button

**Desktop:**
- Height: 72px
- Padding: 16px 32px
- Title: 2.5rem (40px)
- Logout: 44x44px icon button

---

### Message Bubbles

#### User Messages
**Dimensions:**
```
┌─────────────────────────┐
│ User Message            │ ← 18px border-radius (top-left)
│                         │ ← 18px border-radius (top-right)
│ Font: 16px              │
│ Padding: 12px 16px      │
│ Line-height: 1.4        │ ← 4px border-radius (bottom-right)
└─────────────────────────┘ ← 18px border-radius (bottom-left)

Mobile:   85% max-width
Tablet:   75% max-width
Desktop:  70% max-width
```

**Colors:**
- Background: #0969da (primary-blue)
- Text: #ffffff (white)
- Shadow: 0 2px 8px rgba(0, 122, 255, 0.25)

#### Assistant Messages
**Dimensions:**
```
┌│────────────────────────┐
││ Assistant Message      │ ← 18px border-radius (all corners)
││                        │ ← except bottom-left = 4px
││ Font: 16px             │
│  Padding: 16px 20px    │
│  Line-height: 1.5      │
└────────────────────────┘

3px blue border on left side

Mobile:   90% max-width
Tablet:   80% max-width
Desktop:  75% max-width
```

**Colors:**
- Background: #161b22 (bg-secondary)
- Text: #f0f6fc (text-primary)
- Border-left: 3px solid #0969da
- Border: 1px solid #30363d
- Shadow: 0 1px 3px rgba(0, 0, 0, 0.2)

---

### Input Bar

**Mobile Layout:**
```
┌─────────────────────────────────┐
│ Padding: 8px 16px               │
│ ┌────────────────────────┬────┐ │
│ │                        │ [⬆]│ │
│ │   Input Field (56px)   │48px│ │
│ │   Border-radius: 28px  │    │ │
│ │   Font: 16px           │    │ │
│ └────────────────────────┴────┘ │
│ Safe-area-inset-bottom          │
└─────────────────────────────────┘
Total: 72px + safe-area
```

**Desktop Layout:**
```
┌──────────────────────────────────────────┐
│        Centered (800px max-width)        │
│ Padding: 12px 32px                       │
│ ┌───────────────────────────────┬────┐  │
│ │                               │ [⬆]│  │
│ │   Input Field (56px)          │48px│  │
│ │   Border-radius: 28px         │    │  │
│ │   Font: 17px                  │    │  │
│ └───────────────────────────────┴────┘  │
└──────────────────────────────────────────┘
```

**Input Field:**
- Height: 56px (min), 120px (max when expanded)
- Border-radius: 28px (pill shape)
- Padding: 14px 60px 14px 16px
- Font-size: 16px (prevents iOS zoom)
- Background: #21262d (bg-tertiary)
- Border: 1px solid #30363d (border-light)
- Focus border: 2px solid #0969da
- Focus shadow: 0 0 0 3px rgba(0, 122, 255, 0.15)

**Send Button:**
- Size: 48x48px (circular)
- Position: Absolute right 12px, centered vertically
- Background: #0969da (primary-blue)
- Icon: 24x24px
- Shadow: 0 2px 8px rgba(0, 122, 255, 0.3)
- Active scale: 0.92

---

## Spacing System

**Base Unit:** 8px

**Spacing Scale:**
```
4px   - Tight spacing (element gaps)
8px   - Small spacing (text margins)
12px  - Medium spacing (message margins)
16px  - Default spacing (mobile padding)
24px  - Large spacing (tablet padding)
32px  - Extra large (desktop padding)
48px  - Huge spacing (section gaps)
```

**Applied Spacing:**
- Message vertical margin: 12px
- Message horizontal margin: 16px
- Input bar padding: 8px (mobile), 12px (desktop)
- Container padding: 16px (mobile), 24px (tablet), 32px (desktop)
- Section gaps: 16px
- Header margin-bottom: 4px (title to subtitle)

---

## Typography

### Font Family
```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont,
             'SF Pro Display', 'Helvetica Neue', sans-serif;
```

### Size Scale

**Mobile (767px-):**
```
h1: 1.75rem (28px) - line-height: 1.2
h2: 1.5rem (24px)  - line-height: 1.25
h3: 1.25rem (20px) - line-height: 1.3
body: 1rem (16px)  - line-height: 1.4
caption: 0.875rem (14px)
```

**Desktop (1024px+):**
```
h1: 2.5rem (40px)     - line-height: 1.05
h2: 2rem (32px)       - line-height: 1.1
h3: 1.75rem (28px)    - line-height: 1.2
body: 1.0625rem (17px) - line-height: 1.47
caption: 0.9375rem (15px)
```

### Font Weights
- Light: 300
- Regular: 400 (body text, messages)
- Medium: 500 (labels)
- Semibold: 600 (headings, buttons)
- Bold: 700 (titles)

---

## Color System

### Backgrounds
```
--bg-primary:    #0d1117  (main app background)
--bg-secondary:  #161b22  (cards, messages)
--bg-tertiary:   #21262d  (input fields)
--bg-accent:     #30363d  (hover states)
--bg-elevated:   #1c2128  (modals, overlays)
```

### Text
```
--text-primary:    #f0f6fc  (headings, important text)
--text-secondary:  #c9d1d9  (body text)
--text-muted:      #8b949e  (placeholders, captions)
--text-accent:     #7d8590  (metadata)
--text-inverse:    #0d1117  (text on light backgrounds)
```

### Borders
```
--border-light:   #30363d  (subtle borders)
--border-medium:  #444c56  (standard borders)
--border-accent:  #586069  (focused borders)
--border-strong:  #6e7681  (prominent borders)
```

### Interactive
```
--primary-blue:       #0969da  (buttons, links)
--primary-blue-dark:  #0860ca  (hover state)
--secondary-blue:     #58a6ff  (accents)
--success-green:      #3fb950  (success states)
--warning-orange:     #d29922  (warnings)
--error-red:          #f85149  (errors)
```

---

## Shadows

### Message Bubbles
```css
/* User messages */
box-shadow: 0 2px 8px rgba(0, 122, 255, 0.25);

/* Assistant messages */
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
```

### Input Bar
```css
/* Container */
box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);

/* Input field default */
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);

/* Input field focused */
box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15);

/* Send button */
box-shadow: 0 2px 8px rgba(0, 122, 255, 0.3);
```

---

## Animations

### Message Slide In
```css
@keyframes messageSlideIn {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

Duration: 250ms
Easing: cubic-bezier(0.25, 0.46, 0.45, 0.94)
```

### Button Press
```css
/* Active state */
transform: scale(0.92);
transition: all 150ms ease-out;
```

### Focus Glow
```css
/* Input focus */
transition: all 200ms ease-out;
box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15);
```

### Hover Effects (Desktop only)
```css
/* Send button hover */
transform: scale(1.05);
transition: all 150ms ease-out;
```

---

## Accessibility

### Touch Targets
- Minimum: 44x44px
- Send button: 48x48px
- Logout button: 44x44px
- Input field: 56px height

### Focus Indicators
```css
/* Keyboard focus */
outline: 2px solid #0969da;
outline-offset: 2px;
box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.2);

/* Mobile focus */
outline-width: 3px;
outline-offset: 3px;
```

### Color Contrast Ratios
- User messages: 7:1 (white on blue)
- Assistant messages: 7:1+ (primary on secondary)
- Body text: 7:1+ (secondary on primary)
- All text meets WCAG 2.1 AA standards

---

## Responsive Behavior

### Breakpoint Transitions
```
320px ────────▶ 767px : Mobile
                  ▼
768px ────────▶ 1023px : Tablet
                  ▼
1024px ───────▶ 1439px : Desktop
                  ▼
1440px ───────▶ 1919px : Large Desktop
                  ▼
1920px+ ──────────────▶ Ultra-wide
```

### Adaptive Elements

**Message Width:**
- Mobile: 85%/90%
- Tablet: 75%/80%
- Desktop: 70%/75%
- Large: 65%/70%

**Input Bar Position:**
- Mobile: Full width, fixed bottom
- Tablet+: Centered with max-width

**Typography:**
- Mobile: 16px base
- Tablet: 17px base
- Desktop: 17px base
- Large: 18px base

**Padding:**
- Mobile: 16px
- Tablet: 24px
- Desktop: 32px
- Large: 48px

---

## Implementation Notes

### Critical CSS Rules
1. Input font-size MUST be 16px on iOS to prevent zoom
2. Touch targets MUST be minimum 44x44px
3. Fixed positioning MUST use safe-area-inset for iOS
4. Focus-visible SHOULD be used for keyboard vs mouse distinction

### Performance Considerations
1. Avoid backdrop-filter on mobile (heavy GPU operation)
2. Use simple box-shadows (max 2 layers)
3. Limit animation duration to 250ms or less
4. Use CSS containment on repeated elements

### Browser Compatibility
- CSS Grid: 96%+
- Custom Properties: 95%+
- focus-visible: 92%+ (graceful fallback)
- env(safe-area-inset): iOS 11+
