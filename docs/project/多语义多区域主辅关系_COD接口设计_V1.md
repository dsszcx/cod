# 多语义—多区域—主辅关系—COD：接口设计 V1

当前数据范围（2026-10-26）：用户要求后续仅考虑伪装图像，train/val/test均排除NonCAM。新清单、数量、服务器数据视图与比较限制见[CAM-only数据范围调整](CAM_only数据范围调整_20261026.md)。不改变本设计的多语义、多区域、弱角色与COD最终判断原则。

日期：2026-10-22。状态：区域打包、几何关系和batch接口已实现并通过CPU/服务器数据验收；最小关系网络已实现并通过本地/服务器CPU前向与合成梯度测试，以及真实9张图的前向验收。未训练或验证性能。

2026-10-26更新：上述为10月22日历史状态。现已完成共用训练器、64/16图三组各10epoch短训练，以及复现/零残差控制；尚无正式性能结论。当前先验残差幅值很小，关闭先验几乎不改变预测，不能把验证MAE差异归功于关系模块。详见[本周短训练与复现性诊断](../experiments/本周短训练与复现性诊断_20261026.md)。本轮两先验组设置`use_node_geometry=False`，避免直连对照预先引入成对几何均值；加载本轮权重时需保留此设置。

## 1. 已确认主线及边界

原始RGB → 多语义标签 → 多候选区域 → 可学习的主辅关系 → COD最终预测。

Qwen提供语义与弱关系建议，SAM3提供区域；都不决定最终主目标。角色可错、关系可缺失；COD可修正或忽略先验。部件既可支持整体候选，也可直接帮助最终预测。环境不是固定负样本。

本设计替代的是后续新方法的接口，不重写或删除历史E0–E4、旧8通道流程及其结果。新方法不依赖selection/V2/V3的唯一主目标选择，不用它们的失败评分充当监督标签。当前9张测试样本仅作开发素材，不转为训练集或正式性能证明。

## 2. 代码检查结论与接入位置

- `Network_SAM3.py`现有接口：`forward(x, target, aux, neg_flag)`，骨干输入8通道；先验融合还在RGB输入前调制像素。这不是新设计要求的可变区域集合接口。
- 其通道为RGB3+target1+aux3+neg_flag1。历史第8通道是neg_flag，不是额外语义。`dataloader_sam3.py`包含按NonCAM文件名生成neg_flag的模式，新方法不采用这个输入或文件名类别信息。
- `Network_PVTv2.py`具备RGB骨干、三尺度RFB、NCD及ReverseStage，可以作为新网络基础。
- 建议新建独立网络与加载器，不改变旧入口；使用3通道骨干，在RFB输出与NCD/ReverseStage之间引入残差关系分支。下游统一使用融合后的特征，不只修改NCD而遗漏反向细化路径。
- 计划接口：`forward(rgb, region_masks, region_valid, role_hint, role_hint_valid, relation_geom, relation_valid)`；这些是拟定参数，不是已存在的可调用API。
- 如初始化自既有baseline，应完整复制兼容的RGB/解码器权重并核对键与形状。已知740键baseline对应channel=64、use_ca=False；新分支另行初始化，不能把整网部分加载伪装为严格加载。

## 3. 存储格式：区域不提前变成唯一答案

每张图片一个目录，以manifest引用原始图片，不在网络特征或Qwen提示中传文件名/分组。

```text
<sample>/
  regions.npz       # 数值数组，无pickle/object数组
  regions.json      # 标签、候选ID、来源、状态、坐标变换
```

原始候选PNG及旧日志不删除。转换产物另存，不修改来源。

### 3.1 regions.npz字段

| 字段 | 单样本形状/类型 | 含义 |
|---|---|---|
| masks | [N,Hm,Wm] uint8，0/1 | 同一图像坐标下的候选Mask，不合并成一个目标 |
| role_hint | [N,4] float32 | whole / part / context / unknown弱角色提示 |
| role_hint_valid | [N] bool | 是否有角色来源；非概率或可靠性分数 |
| sam3_score | [N] float32 | SAM3原始置信度，供报告，不默认用于融合权重 |
| relation_geom | [N,N,8] float32 | 有方向的几何证据，顺序定义见下 |
| relation_valid | [N,N] bool | 两节点均有效且i≠j；不是“关系为真” |

N允许为0。存储时不补齐；批处理时再补齐并生成region_valid。role_hint可用one-hot记录标签生成阶段的提议；unknown对应独立位置，不能把缺失暗示为context。它不是Qwen给出的四类概率。旧标签的object可映射whole，但仅是弱提示，不锁定角色。

动作只保留文本，不伪造Mask或空间节点。候选生成为空、格式失败、没有可分割标签，分别写入JSON状态；即使都导致N=0，也不解释为图中没有目标。

### 3.2 regions.json字段

- schema_version、image_sha256、original_hw、mask_hw、坐标缩放方式、源候选记录及哈希。
- 每节点candidate_id、tag_id、原始英文tag、proposed_kind、mask_index、生成来源和状态。
- 原始语义提示来源，不人工改成“正确类别”；同一Mask对应多个tag时保留各自来源。
- candidate_generation_status与失败原因；无GT字段，不读取文件名中的类别来建立标签。

第一版的语义通过“标签生成的Mask＋弱角色提示”进入网络；标签文本完整存储，但暂不声称已经把文本嵌入网络。显式文本编码器属于后续独立设计/消融，不能用字符串哈希代替有语义的向量。

## 4. 三类关系及几何证据

关系含义为：部件—整体支持、环境—对象关联、重复/竞争。当前只计算证据，不依据阈值写死“属于/不属于”。所有有效节点对都可进入关系分支，unknown节点不被排除。

设Ai、Aj为区域面积，I为交集、U为并集；坐标归一化到图像宽高。每个有向节点对(i,j)的8项证据按以下固定顺序存储：

1. IoU = I/U。
2. contain_i_in_j = I/Ai，表示i有多少落在j内。
3. contain_j_in_i = I/Aj。
4. area_log_ratio = tanh(log(Ai/Aj))，保留相对大小方向，范围[-1,1]。
5. dx = (中心x_j−中心x_i)/W。
6. dy = (中心y_j−中心y_i)/H。
7. bbox_gap = 两个轴对齐外接框间最短距离/图像对角线；重叠或接触时为0，非Mask边界距离。
8. near_duplicate = 1[IoU>0.9]，仅重复提示，不自动删除区域。

有效区域面积必须大于0。对角线/补齐位置全部置0并标为无效，不对空Mask求对数。自节点证据由节点自身路径处理，不假装是另一个辅助区域。

部件—整体和环境—对象的实际关系强度由可学习模块结合视觉、弱角色与上述几何决定。三个关系头可独立输出权重，不做互斥的三选一；一对区域可能没有有用关系。它们是关系假设，不是有真值保证的语义关系概率。

第一版不再要求Qwen为每个节点对评分。不因角色不符强制删除候选，也不使用V3的60分准入规则。关系强度不能由GT/测试集标签生成。

## 5. 批处理与几何增强

批内补齐到Nmax；输入张量为：

```text
rgb               [B,3,H,W]
region_masks      [B,Nmax,H,W]
region_valid      [B,Nmax]
role_hint         [B,Nmax,4]
role_hint_valid   [B,Nmax]
relation_geom     [B,Nmax,Nmax,8]
relation_valid    [B,Nmax,Nmax]
```

N=0的样本可用全无效占位槽参与batch，但其占位不是一个真实区域。补齐不得进入pooling、关系归一化或汇聚。开发批次先保留全部候选，不在接口里默默截断；扩到大数据时另行确定并记录候选上限及截断规则。

RGB、GT及所有候选必须使用同一翻转、旋转、裁剪参数。二值Mask采用最近邻几何变换；变换后重新计算几何与有效性，被裁空节点标为无效。不能直接沿用增强前dx、dy、包含关系。颜色增强只用于RGB，不改变Mask。

训练GT独立交给loss，不进入先验生成器或forward接口。图片索引/分组用于文件配对与报告，不转为输入通道。

## 6. 最小网络结构

以3通道RGB backbone为基础。第一版先在1/8尺度的RFB特征F8接入关系模块，1/16、1/32特征不改，保持改动可控；多尺度关系融合之后再比较，不一次加入所有复杂模块。

### 6.1 区域表征

将候选Mask按面积方式下采样为特征网格上的覆盖率，不用最近邻再把小部件直接抹掉。对F8做带覆盖率的masked average pooling，得到每区域视觉token；拼接弱角色编码及几何摘要，经MLP生成区域token。分母按有效覆盖率归一化，并处理零覆盖。

sam3_score不默认充当注意力门控；可在后续单独试验其增量。初始化不依赖Qwen的失败评分。

### 6.2 主辅关系分支

对每个有效(i,j)，输入两节点token与8项几何，得到三类关系权重。对节点i汇聚其他节点的信息，再通过残差更新i。不要硬指定只有whole节点能接受消息；否则早期错误角色会把正确整体永久排除。

重复关系用于学习降低冗余或竞争，不机械设为负数扣分；部件和环境的效果由图像条件下的可学习变换决定。最终只有分割loss监督时，不能声称三个头天然具有可解释的真实关系语义，需要消融和可视化检验。

### 6.3 两条辅助作用路径及RGB残差

- 直接路径：原始区域token投影回空间，部件不依赖某个whole候选也能起作用。
- 关系路径：更新后的token投影回空间，带入候选间的主辅关系。
- 像素/区域门控结合RGB特征独立学习，不强制所有候选权重总和为1；允许全部被忽略。
- 融合：F8_fused = F8 + alpha_direct×P_direct + alpha_relation×P_relation，两个残差系数初始为0。
- F8_fused送入NCD与RS3，其余尺度沿用RGB特征；最终预测不与候选Mask取交集，不限制在候选并集内。

无有效先验时，P_direct与P_relation必须严格为0。禁用分支/无先验的初始模型应与同权重RGB baseline一致；不是在8通道骨干后面简单补0来宣称存在独立RGB路径。

## 7. 实现顺序和验收

第一实施单元已完成：`tools/semantic_prior/pack_regions.py`负责候选打包，`lib/region_relations.py`负责几何、包读取、同步crop/flip/resize及batch接口，`tests/test_region_relations.py`为CPU测试。当前同步变换接口不含旋转，也尚未接入训练GT；后续训练加载器必须同步实现全部空间变换。

先用现有9张候选做接口检查：源候选非空Mask合计37份，应全部保留，N逐图为2、2、8、0、3、3、6、6、7。没有候选的图片也须有单独的合法空包。不再调用Qwen或SAM3。

首批验收项目：

- Mask、候选ID、RGB哈希配对；保存/读取不改变区域含义。
- 小部件被整体包含时，contain_i_in_j接近1，即使IoU小。
- 互换i,j时两个包含率互换，dx/dy及log面积比变号，IoU/间距/重复项不变。
- 空包、全图、相同Mask、相离区域，以及补齐槽均能正确处理。
- 同步裁剪/翻转后关系按新坐标重算，裁空节点不产生NaN。
- 不根据文件名、NonCAM标记、GT或Qwen最终评分过滤节点。

第二实施单元再做关系网络的随机输入前向、梯度、置换一致性、空先验baseline等价测试，不启动完整训练。

训练另行安排：相同RGB baseline、区域直连、直连+关系、不同关系类型的消融；需要独立训练/验证/测试划分。运行通不代表效果提升，权重未训练时也不应要求新关系分支优于baseline。

## 8. 第一实施单元验收结果

2026-10-22在本地Python3.9及服务器cx1 Python环境通过CPU回归；服务器脚本`tools/server/semantic_prior_pack_v1.sh`退出0，结束标志REGION_INTERFACE_V1_DONE。GPU通过CUDA_VISIBLE_DEVICES空值禁用，无Qwen/SAM3调用或训练。

- 9张图全部有合法包，节点数为[2,2,8,0,3,3,6,6,7]，合计37。
- 原始分辨率Mask逐像素等于源PNG；保存/读取的全部数值数组一致，原始候选、标签及旧选择产物均未覆盖。
- 8张HAS_REGIONS，1张NO_CANDIDATES；空样本不是“无目标”标签。
- 174个有向有效节点对只是几何证据，不是174条已确认的语义关系。
- 试验缩放到352×352后37个区域均仍非空；本批未观察到丢失，不保证任意未来小目标都不会在最近邻缩放时消失。
- 批内rgb为[9,3,352,352]，region_masks为[9,8,352,352]，relation_geom为[9,8,8,8]；补齐不进入有效节点或关系，所有数值有限。
- 测试覆盖方向性包含率、几何对称/反对称、节点置换、全图/重复/相离/空Mask、全空batch、翻转、裁空、同步缩放、源候选保留与NPZ读取。

服务器产物：`/srv/cod-project/outputs/SEMANTIC_PRIOR_PILOT_V1/region_packages_v1/`。其中PACKAGE_SUMMARY.json为打包报告，PACKAGE_BATCH_CHECK.json为逐像素及batch验收报告，PACKAGE_MANIFEST.csv定位逐图regions.npz/regions.json。

实际读取接口：`load_region_package(path)`返回数值区域数据及region_valid；调用方加入对齐的CHW RGB后，可用`transform_region_sample`同步crop/flip/resize并重算关系，再由`collate_region_samples`返回计划forward需要的七个张量。SAM3分数仅存储，不进入模型batch；GT和字符串元数据不进入该字典。

下一单元为最小关系网络和前向/梯度验收，仍不自动开始训练。

## 9. 最小关系网络实现

新增`Network_Relation.py`及`lib/region_fusion.py`，旧RGB和8通道网络文件不变。新网络继承RGB模块结构/键名，在1/8尺度RFB后融合，两条残差系数初始0；融合特征同时用于NCD和RS3。

区域token输入为masked-pooling视觉特征、4维弱角色、角色有效位、8维邻接几何均值及区域覆盖比例。Mask到特征尺度使用area插值。三头软关系不按角色硬筛选，所有有效有向区域对均可参与；每个头独立sigmoid而非全候选softmax。三头语义未受独立监督，不能仅凭命名声称已学会真实部件/环境关系。

关系消息按`1+关系权重和`归一化，避免仅有一个辅助时被迫贡献全部权重。关系路径投影更新token与自身token的差值，不重复投影全部自身信息；没有有效区域对时关系图严格0，单个区域仍能走直连路径。空间回投按`1+覆盖率和`归一化，不按可学习门控的和归一化，因此门控可共同趋零。重复关系仍是可学习假设，不先删除相同Mask。

`load_rgb_state_dict`先检查全部RGB键和形状，再复制；只排除显式新增的region_fusion参数。不是静默跳过不匹配键。默认channel=64、use_ca=False，与已知740键RGB权重对应。

本地`tests/test_region_network.py`通过：

- 关系模块及完整网络前向均有限，完整网络返回4份预测。
- 零初始化残差系数时，输出与相同权重RGB网络逐元素完全相同。
- 临时激活系数后，空先验仍完全等于RGB，非空先验能够影响特征。
- 候选重排后融合结果不变（允许浮点加法误差），无效补齐槽不影响输出。
- 缺失角色提示不排除节点；单像素部件通过area pooling保留非零覆盖率。
- 无节点/全无效节点及无有效边时无NaN；无边的关系路径为0、直连仍可用。
- 零系数时首次反向只允许系数获得有效梯度，内部先验分支梯度为0，这是零初始化的预期行为。将系数临时设非零后，节点、边、消息、投影和门控均有有限非零梯度。
- 完整网络的骨干、解码器和关系边编码器均有有限非零梯度。
- RGB740键完整匹配；缺键/首层形状错误均被拒绝。

梯度验收只使用合成输入、eval模式和一次backward，不创建optimizer，不调用step，不更新权重、不导出checkpoint，不是训练实验。

服务器验收入口为`tools/server/semantic_prior_network_check_v1.sh`，先运行同样CPU测试，再由`tools/semantic_prior/check_region_network.py`读取真实9张原图/候选包及原baseline权重。禁用GPU，不读GT，生成`region_network_check_v1/NETWORK_CHECK.json`。临时alpha=0.1只验证线路连通，不能用随机初始化分支的差异宣称性能提升。

### 服务器实际验收

脚本退出0，结束标志REGION_NETWORK_V1_DONE。严格加载740/740键，原权重SHA256为f4083aebb5053cef8545a4064ce19675cefbfac20300442847f4142bb2647707，运行前后未变。9张原图、37个区域都完成352×352前向，每张4份[1,1,352,352]输出，数值均有限。

9张的零系数输出相对baseline最大logit差值均为0；唯一空先验样本在alpha=0.1时差值仍0。其余图临时激活未训练分支后的最大logit差值约0.0153–0.0461，仅用于证明分支连通，不是准确率提升。未读GT、未保存预测指标、未更新或导出权重。验收报告已取回本地outputs/SEMANTIC_PRIOR_PILOT_V1/region_network_check_v1/NETWORK_CHECK.json。

下一步需单独安排训练加载器、训练/验证划分、训练集先验准备及小规模消融；不能把这9张测试图用作训练数据。当前网络实现并未授权自动开始训练。
