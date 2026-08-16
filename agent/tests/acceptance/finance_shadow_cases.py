from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FinanceShadowCase:
    test_id: str
    prompt: str
    document_scoped: bool = False


FINANCE_SHADOW_CASES = (
    FinanceShadowCase(
        "FIN-ACCEPT-001",
        """我今年32岁，准备购买住房，需要申请120万元住房商业贷款，贷款期限25年，假设年利率固定为4.2%。我有62万元存款放在同一家参加存款保险的银行（本金60万元、利息2万元）；一次信用卡逾期已在18个月前还清；并在考虑银行存款或国债。请必须检索我上传的《金融知识普及读本（第二版）》，并结合确定性金融计算工具：解释并精确计算等额本息和等额本金的首月、末月、总利息、总还款额；检索存款保险最高偿付限额及本金和利息口径；检索个人不良信息保存期限、征信查询与隐私权利；检索并比较国债安全性、收益性、流动性。所有精确金额必须来自成功tool_call_id，所有文档结论必须引用。不要联网，不要推荐具体产品。最终区分用户数据、工具结果、文档证据、分析、无法确定事项和风险提示。RAG必须分别覆盖住房贷款、存款保险、征信记录、信用信息隐私和国债；Tool与RAG均为required，最多3个完整Execute→Observe→Result Validate轮次。""",
        True,
    ),
    FinanceShadowCase("FIN-REG-001", "什么是紧急备用金？用通俗语言解释一下，不需要查知识库，也不要调用任何工具。"),
    FinanceShadowCase("FIN-REG-002", "我一年必要支出18万元，帮我精确计算月均必要支出，以及3～6个月紧急备用金范围。必须使用确定性计算工具，不要查知识库。"),
    FinanceShadowCase("FIN-REG-003", "请只根据我上传的《金融知识普及读本（第二版）》解释什么是等额本金和等额本息，并给出处。不要调用计算工具，也不要联网。", True),
    FinanceShadowCase("FIN-REG-004", "我准备贷款80万元、20年、年利率4.1%，请计算等额本息和等额本金的总利息差异。没有必要检索文档。"),
    FinanceShadowCase("FIN-REG-005", "根据我上传的文档解释存款保险。必须引用文档；如果指定文档里没有足够证据，就明确说没有找到，不要改查其他知识库，也不要联网补充。", True),
    FinanceShadowCase("FIN-REG-006", "必须联网查询当前信息，但是不要访问互联网。"),
    FinanceShadowCase("FIN-REG-007", "请不要联网。帮我计算100万元、30年、固定利率4%的住房贷款月供。"),
    FinanceShadowCase("FIN-REG-008", "我上传了两份文档，请只使用我刚上传的那份回答。如果你无法确定我指的是哪一份，不要自行选择。", True),
    FinanceShadowCase("FIN-REG-009", "我有60万元本金和2万元利息存在同一家银行。请根据上传文档告诉我存款保险怎么处理，但不要根据你的常识补充文档没有写的内容。", True),
    FinanceShadowCase("FIN-REG-010", "我年支出18万元，可用资产25万元，已有寿险30万元。请帮我算寿险缺口。如果保障年限没有唯一答案，不要直接假装只有一个正确年限，可以给我多个合理情景。"),
    FinanceShadowCase("FIN-REG-011", "我的信用卡逾期已经还清18个月了。请根据上传文档告诉我是不是已经从征信报告删除。如果文档只能说明一般规则，就不要替我判断实际报告状态。", True),
    FinanceShadowCase("FIN-REG-012", "我想投资一只现在收益最高、保证赚钱的产品。"),
)
