import numpy as np

class RewardCalculator:
    def __init__(self, w1=0.40, w2=0.25, w3=0.15, w4=0.05, w5=0.15,
                 tc_rate=0.001, eta=0.01, rf=0.0001):
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.w4, self.w5 = w4, w5
        self.tc_rate = tc_rate
        self.eta = eta
        self.rf = rf
        self.A = 0.0
        self.B = 0.0
        self.market_rets = []
        self.port_rets = []
        self.window = 20

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self.market_rets = []
        self.port_rets = []

    def compute(self, pv, prev_pv, peak, trade_executed,
                market_return=0.0) -> float:
        r_t = np.log(pv / prev_pv) if prev_pv > 0 else 0.0

        # Component 1: Log Return
        c1 = r_t

        # Component 2: Sortino-style downside risk
        c2 = -max(0.0, -r_t) ** 2

        # Component 3: Differential Sharpe Ratio (Moody & Saffell)
        A_prev, B_prev = self.A, self.B
        self.A = A_prev + self.eta * (r_t - A_prev)
        self.B = B_prev + self.eta * (r_t**2 - B_prev)
        var = self.B - self.A**2
        if var > 1e-8:
            dA = r_t - A_prev
            dB = r_t**2 - B_prev
            num = B_prev * dA - 0.5 * A_prev * dB
            c3 = float(np.clip(num / (var**1.5), -1.0, 1.0))
        else:
            c3 = 0.0

        # Component 4: Transaction cost penalty
        c4 = -self.tc_rate if trade_executed else 0.0

        # Component 5: Treynor ratio contribution
        self.port_rets.append(r_t)
        self.market_rets.append(market_return)
        if len(self.port_rets) > self.window:
            self.port_rets.pop(0)
            self.market_rets.pop(0)
        c5 = 0.0
        if len(self.port_rets) >= 5:
            p = np.array(self.port_rets)
            m = np.array(self.market_rets)
            mvar = np.var(m)
            if mvar > 1e-8:
                beta = np.cov(p, m)[0, 1] / mvar
                beta = np.clip(beta, 0.1, 3.0)
                c5 = float(np.clip((r_t - self.rf) / beta, -0.5, 0.5))

        return float(self.w1*c1 + self.w2*c2 + self.w3*c3 +
                     self.w4*c4 + self.w5*c5)
