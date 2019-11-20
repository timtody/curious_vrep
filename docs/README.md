---
title: Blogging Like a Hacker
lang: en-US
---

[[toc]]


# Documentation

Welcome on some documentation

## Some heading

### Yet another heading

# Big heading again

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


:tada: :100:


::: tip
This is a tip
:::

::: warning
This is a warning
:::

::: danger
This is a dangerous warning
:::


``` py{9}
class Transition:
    def __init__(self):
        self.state_old = None
        self.state_new = None
        self.action = None
        self.reward = None

    def set_state_old(self, state_old):
        self.state_old = state_old

    def set_state_new(self, state_new):
        self.state_new = state_new

    def set_action(self, action):
        self.action = action

    def set_reward(self, reward):
        self.reward = reward
```

<<< @/.gitignore{highlightLines}


1 + 1 is {{ 1+1 }}

<span v-for="i in 3">{{ i }} </span>

<doc-index/>