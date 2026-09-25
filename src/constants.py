"""Global physical and tensor-layout constants for the basketball diffusion project.

Court frame: **basket/rim at the origin (0, 0)**, x-axis pointing toward half-court,
y-axis lateral. All lengths in meters, time in seconds.
"""

# ---------------------------------------------------------------------------
# Court geometry (half-court, basket-referenced frame)
# ---------------------------------------------------------------------------
RIM_RADIUS = 0.2286          # 18-inch diameter rim -> radius in meters
HOOP_X = 0.0                 # rim x (origin)
HOOP_Y = 0.0                 # rim y (origin)
BACKBOARD_X = -0.375         # backboard sits ~37.5 cm behind the rim center
COURT_LENGTH_HALF = 14.0     # half-court length (FIBA ~14 m; NBA 14.0 m from rim to half line ~ 14.0)
COURT_WIDTH = 15.24          # 50 ft baseline width in meters
THREE_PT_RADIUS = 7.24       # NBA 3-pt arc radius (m) (23'9")
THREE_PT_CORNER_Y = 6.7      # 22 ft corner-line |y| cutoff (m)
RESTRICTED_RADIUS = 1.25     # 4 ft restricted-area arc (m)
FT_CIRCLE_RADIUS = 1.8       # free-throw circle radius (m)
FT_LINE_X = 5.8              # free-throw line distance from baseline/rim origin (m)  (19 ft from rim -> ~4.6 m... NBA FT line 15 ft from backboard)
PAINT_WIDTH = 4.88           # 16 ft lane width (m)
PAINT_LENGTH = 5.79          # 19 ft from baseline (m)

# ---------------------------------------------------------------------------
# Agents & features
# ---------------------------------------------------------------------------
N_AGENTS = 11                # 10 players + 1 ball
N_PLAYERS = 10
BALL_IDX = 10                # slot of the ball inside the agent axis
N_ROLES = 5                  # PG/SG/SF/PF/C slots (offense always 1..5)
FEAT_DIM = 4                 # x, y, vx, vy per agent
SEQ_LEN = 100               # frames per sample (4 s @ 25 fps)

# Role semantic indices along the agent axis:
#   0..4   -> offense roles 1..5
#   5..9   -> defense
#   10     -> ball
OFFENSE_SLICE = slice(0, 5)
DEFENSE_SLICE = slice(5, 10)

# ---------------------------------------------------------------------------
# Physics limits (biomechanically plausible ranges)
# ---------------------------------------------------------------------------
V_MAX_PLAYER = 7.5           # m/s, ~ fastest NBA players top out near this
V_MAX_BALL_PASS = 24.0       # m/s, hard passes
V_MAX_BALL_DRIBBLE = 8.0     # m/s dribble/handling speeds
V_MAX = [V_MAX_PLAYER] * N_PLAYERS + [V_MAX_BALL_PASS]
A_MAX = 5.0                  # m/s^2 sustained-acceleration cap used by the loss
R_MIN = 0.6                  # minimum center distance before overlap penalty (m)
