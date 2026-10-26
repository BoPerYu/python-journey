class StudyTask:
    def __init__(self, title, minutes):
        self.title = title
        self.minutes = minutes
        self.done = False
    def finish(self):
        self.done=True
    def summary(self):
        if self.done:
            return f"{self.title} | {self.minutes}min | 完成"
        else:
            return f"{self.title} | {self.minutes}min | 未完成"

task1 = StudyTask("LeetCode", 30)
task2 = StudyTask("C++", 40)

task1.finish()

print(task1.summary())
print(task2.summary())
print(task1 is task2)