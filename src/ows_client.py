"""
Minimal synchronous client for the goszakup OWS v3 GraphQL API.

Adapted from Tender Radar's async ETL client (backend/app/etl/client.py), stripped down
to what a research pipeline needs: one `graphql()` call and `limit/after` pagination
with retries. No database, no async.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

# OWS v3 caps page size at 200 for most entities.
MAX_PAGE_SIZE = 200
RETRY_STATUSES = {429, 500, 502, 503, 504}


class OWSError(RuntimeError):
    pass


class OWSClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        delay: float = 0.3,
        max_retries: int = 6,
        timeout: float = 90.0,
    ):
        self.token = token or os.environ.get("OWS_TOKEN")
        self.base_url = (base_url or os.environ.get("OWS_BASE_URL") or "https://ows.goszakup.gov.kz").rstrip("/")
        if not self.token:
            raise OWSError("OWS_TOKEN is not set (put it in .env)")
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    # ── low level ────────────────────────────────────────────────────────────

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST one GraphQL query. Returns the full JSON (data + extensions). Retries transient errors."""
        payload = {"operationName": None, "query": query, "variables": variables or {}}
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.session.post(f"{self.base_url}/v3/graphql", json=payload, timeout=self.timeout)
                if resp.status_code in RETRY_STATUSES:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                body = resp.json()
                if "errors" in body:
                    msg = body["errors"][0].get("message", str(body["errors"]))
                    raise OWSError(f"GraphQL error: {msg}")
                return body
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt >= self.max_retries:
                    raise OWSError(f"giving up after {attempt} attempts: {e}") from e
                wait = min(2 ** attempt, 60)
                log.warning("OWS transient error (%s), retry %d/%d in %ds", e, attempt, self.max_retries, wait)
                time.sleep(wait)

    # ── pagination ───────────────────────────────────────────────────────────

    def paginate(
        self,
        query: str,
        data_key: str,
        variables: dict[str, Any] | None = None,
        page_size: int = MAX_PAGE_SIZE,
        after: int = 0,
        max_records: int = 0,
    ) -> Iterator[tuple[list[dict], int]]:
        """
        Yield (batch, last_id) pages using OWS `limit/after` cursor pagination.
        `after` lets a caller resume from a checkpoint. `max_records=0` means no cap.
        """
        page_size = min(page_size, MAX_PAGE_SIZE)
        fetched = 0
        while True:
            vars_ = dict(variables or {})
            vars_["limit"] = page_size
            if after:
                vars_["after"] = after
            body = self.graphql(query, vars_)
            batch = body.get("data", {}).get(data_key) or []
            if not batch:
                return
            page_info = body.get("extensions", {}).get("pageInfo", {})
            last_id = page_info.get("lastId") or batch[-1].get("id") or batch[-1]["pid"]
            fetched += len(batch)
            yield batch, last_id
            if not page_info.get("hasNextPage", len(batch) == page_size):
                return
            if max_records and fetched >= max_records:
                return
            after = last_id
            time.sleep(self.delay)

    def total_count(self, query: str, variables: dict[str, Any] | None = None) -> int | None:
        """Cheap count of a filtered query via extensions.pageInfo.totalCount."""
        vars_ = dict(variables or {}); vars_["limit"] = 1
        body = self.graphql(query, vars_)
        return body.get("extensions", {}).get("pageInfo", {}).get("totalCount")


# ── GraphQL templates ────────────────────────────────────────────────────────
# Field lists mirror Tender Radar's templates plus nested reference names
# (RefBuyStatus / RefContractStatus / RefTradeMethods) so the dataset is self-describing.

GQL_TRD_BUY = """
query($limit: Int!, $after: Int, $filter: TrdBuyFiltersInput) {
  TrdBuy(limit: $limit, after: $after, filter: $filter) {
    id numberAnno nameRu nameKz
    refTradeMethodsId refSubjectTypeId refBuyStatusId refTypeTradeId refSpecPurchaseTypeId
    publishDate startDate endDate repeatStartDate repeatEndDate itogiDatePublic
    totalSum countLots
    orgBin orgPid orgNameRu customerBin customerPid customerNameRu
    idSupplier biinSupplier
    singlOrgSign isLightIndustry isConstructionWork disablePersonId parentId
    finYear kato systemId lastUpdateDate indexDate
    RefBuyStatus { nameRu }
    RefTradeMethods { nameRu }
  }
}
"""

GQL_LOTS = """
query($limit: Int!, $after: Int, $filter: LotsFiltersInput) {
  Lots(limit: $limit, after: $after, filter: $filter) {
    id trdBuyId trdBuyNumberAnno lotNumber
    nameRu nameKz descriptionRu
    amount count
    customerId customerBin customerNameRu
    dumping unionLots psdSign consultingServices
    refLotStatusId refTradeMethodsId refBuyTradeMethodsId
    singlOrgSign isLightIndustry isConstructionWork disablePersonId isDeleted
    enstruList plnPointKatoList
    systemId lastUpdateDate indexDate
    RefLotsStatus { nameRu }
  }
}
"""

GQL_TRD_APP = """
query($limit: Int!, $after: Int, $filter: TrdAppFiltersInput) {
  TrdApp(limit: $limit, after: $after, filter: $filter) {
    id buyId supplierId supplierBinIin protId protNumber dateApply systemId indexDate
    AppLots { id lotId statusId price amount discountValue discountPrice }
  }
}
"""

GQL_CONTRACT = """
query($limit: Int!, $after: Int, $filter: ContractFiltersInput) {
  Contract(limit: $limit, after: $after, filter: $filter) {
    id parentId rootId trdBuyId trdBuyNumberAnno
    contractNumber contractNumberSys
    customerBin supplierBiin supplierId
    refContractStatusId refContractCancelId refContractTypeId refContractAgrFormId
    faktTradeMethodsId refFinsourceId isGu enforcement
    contractSum contractSumWnds faktSum faktSumWnds
    crdate signDate planExecDate faktExecDate contractEndDate ecEndDate trdBuyItogiDatePublic
    finYear systemId lastUpdateDate indexDate
    RefContractStatus { nameRu }
    RefContractCancel { nameRu }
  }
}
"""

GQL_RNU = """
query($limit: Int!, $after: Int) {
  Rnu(limit: $limit, after: $after) {
    id pid supplierBiin supplierNameRu startDate endDate systemId indexDate
  }
}
"""

# Registered participants (companies, entrepreneurs, state bodies). Keyed by pid, not id.
# email / phone / website are hashed by the collector before anything is written to disk.
GQL_SUBJECT = """
query($limit: Int!, $after: Int) {
  Subjects(limit: $limit, after: $after) {
    pid bin iin regdate crdate year
    typeSupplier krpCode kseCode okedList refKopfCode katoList countryCode
    markResident markSmallEmployer markNationalCompany markWorldCompany
    markStateMonopoly markNaturalMonopoly markPatronymicProducer markPatronymicSupplyer markAssocWithDisab
    qvazi customer supplier organizer isSingleOrg branches parentSubject
    email phone website lastUpdateDate
  }
}
"""

# Supplier bank details and legal address from contracts (hashed by the collector).
GQL_CONTRACT_BANK = """
query($limit: Int!, $after: Int, $filter: ContractFiltersInput) {
  Contract(limit: $limit, after: $after, filter: $filter) {
    id trdBuyId supplierBiin supplierIik supplierBik supplierLegalAddress
  }
}
"""

# Annual plan points. `Plans` serves only the current version of each point; the version a lot
# was published against (often superseded since, isActive=0) is reachable only via Lots { Plans }.
PLAN_FIELDS = """
    id rootrecordId idc sysSubjectsId sysOrganizatorId subjectBiin subjectNameRu nameRu descRu extraDescRu
    refTradeMethodsId refUnitsCode codeGu count price amount refMonthsId refPlnPointStatusId plnPointYear
    refSubjectTypeId refEnstruCode refFinsourceId refAbpCode isQvazi dateCreate timestamp refPointTypeId
    sum1 sum2 sum3 supplyDateRu prepayment refJustificationId refAmendmentAgreemTypeId refAmendmAgreemJustifId
    contractPrevPointId disablePersonId transferSysSubjectsId transferType refBudgetTypeId
    createdinActId isActive activeActId isDeleted systemId indexDate
    PlanActs { id planActNumber dateApproved planPreliminary refPlanStatusId }
    PlansKato { refKatoCode count }
    PlansSpec { abpCode fkrbProgramCode fkrbSubprogramCode ekrbCode amount }
"""

GQL_PLANS = """
query($limit: Int!, $after: Int, $filter: PlansFiltersInput) {
  Plans(limit: $limit, after: $after, filter: $filter) {%s}
}
""" % PLAN_FIELDS

GQL_LOT_PLANS = """
query($limit: Int!, $after: Int, $filter: LotsFiltersInput) {
  Lots(limit: $limit, after: $after, filter: $filter) {
    id trdBuyId pointList
    Plans {%s}
  }
}
""" % PLAN_FIELDS
