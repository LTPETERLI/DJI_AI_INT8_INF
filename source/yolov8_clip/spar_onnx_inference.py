from ultralytics import SPARYOLO

model = SPARYOLO('/home/peter/服务器文件上传/真实训练文件/train4种缺陷112/weights/best.pt')
model.set_classes(['Broken_strands', 'loose_strands', 'foreign_body_entanglement', 'rust'])

results = model.predict(
    source="/home/peter/PycharmProjects/analogical_exclude/.ultralytics/Broken_strands#0473.jpg",
    imgsz=1024,
    conf=0.05,   # 先放低一点看有没有框
    iou=0.6,
    save=True,
    save_txt=True,
    device=0
)

for r in results:
    print("saved to:", r.save_dir)
