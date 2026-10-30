// Enforces the "Pull Request Requirements" section of CONTRIBUTING.md.
//
// A pull request stays open when any of these hold:
//   - the author is a bot, has write access, or has a commit merged here before
//   - a maintainer labeled the pull request `accepted`
//   - it changes only Markdown files
//   - it closes an issue in this repository labeled with an accepted label
// Anything else is closed with a comment. A pull request the gate closed is
// reopened once it passes: its issue gets an accepted label, a maintainer labels
// it `accepted`, or its author edits it to link an accepted issue.

const ACCEPTED_LABELS = ["accepted", "good first issue", "help wanted"];
const EXEMPT_ASSOCIATIONS = ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR"];
const WRITE_PERMISSIONS = ["admin", "maintain", "write"];
// Hidden markers on the gate's own comments record whether it last closed or
// reopened a pull request.
const CLOSED_MARKER = "<!-- contribution-gate:closed -->";
const REOPENED_MARKER = "<!-- contribution-gate:reopened -->";

const closeMessage = (repo) => `${CLOSED_MARKER}
Thanks for the contribution! Pull requests from first-time contributors need to fix an accepted issue: one a maintainer has labeled \`accepted\`, \`good first issue\` or \`help wanted\`. This pull request doesn't link one, so it has been closed.

Please open or find an issue that describes the problem (for bugs, include a way to reproduce it in a running bot), and link it from this pull request's description with \`Fixes #<issue>\`. This pull request is reopened automatically once the issue is accepted.

See [Pull Request Requirements](https://github.com/${repo}/blob/main/CONTRIBUTING.md#pull-request-requirements) for details.`;

const REOPEN_MESSAGE = `${REOPENED_MARKER}
This pull request has been accepted, so it is open again for review.`;

function hasAcceptedLabel(labels) {
  return labels.some((label) => ACCEPTED_LABELS.includes(label.name));
}

async function isExemptAuthor({ github, context, pr }) {
  if (pr.user.type === "Bot") return true;
  if (EXEMPT_ASSOCIATIONS.includes(pr.author_association)) return true;

  // Private org membership can show up as NONE, so check repository access too.
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      ...context.repo,
      username: pr.user.login,
    });
    return WRITE_PERMISSIONS.includes(data.permission);
  } catch {
    return false;
  }
}

async function isMarkdownOnly({ github, context, pr }) {
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo,
    pull_number: pr.number,
    per_page: 100,
  });
  return files.length > 0 && files.every((file) => file.filename.endsWith(".md"));
}

async function closingIssues({ github, context, pr }) {
  const result = await github.graphql(
    `query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          closingIssuesReferences(first: 10) {
            nodes {
              number
              repository { nameWithOwner }
              labels(first: 20) { nodes { name } }
            }
          }
        }
      }
    }`,
    { ...context.repo, number: pr.number },
  );
  const repo = `${context.repo.owner}/${context.repo.repo}`;
  return result.repository.pullRequest.closingIssuesReferences.nodes
    .filter((issue) => issue.repository.nameWithOwner === repo)
    .map((issue) => ({ number: issue.number, labels: issue.labels.nodes }));
}

async function passesGate({ github, context, pr }) {
  if (await isExemptAuthor({ github, context, pr })) return true;
  if (pr.labels.some((label) => label.name === "accepted")) return true;
  if (await isMarkdownOnly({ github, context, pr })) return true;
  const issues = await closingIssues({ github, context, pr });
  return issues.some((issue) => hasAcceptedLabel(issue.labels));
}

async function wasClosedByGate({ github, context, pr }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: pr.number,
    per_page: 100,
  });
  const last = comments
    .filter((c) => c.body?.startsWith(CLOSED_MARKER) || c.body?.startsWith(REOPENED_MARKER))
    .pop();
  return last !== undefined && last.body.startsWith(CLOSED_MARKER);
}

async function comment({ github, context, pr, body }) {
  await github.rest.issues.createComment({ ...context.repo, issue_number: pr.number, body });
}

async function setState({ github, context, pr, state }) {
  await github.rest.pulls.update({ ...context.repo, pull_number: pr.number, state });
}

async function checkPullRequest({ github, context, core }) {
  const { action, label } = context.payload;
  // Fetch fresh state; the payload can lag behind label changes.
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: context.payload.pull_request.number,
  });

  if (pr.state === "closed") {
    // A gate-closed pull request reopens once it passes: a maintainer labels
    // it `accepted`, or its author edits the description to link an accepted
    // issue.
    const relabeled = action === "labeled" && label.name === "accepted";
    if ((relabeled || action === "edited") && !pr.merged) {
      if (
        (await wasClosedByGate({ github, context, pr })) &&
        (await passesGate({ github, context, pr }))
      ) {
        await setState({ github, context, pr, state: "open" });
        await comment({ github, context, pr, body: REOPEN_MESSAGE });
        core.info(`Reopened #${pr.number}: now passes the gate.`);
      }
    }
    return;
  }

  if (await passesGate({ github, context, pr })) {
    core.info(`#${pr.number} passes the contribution gate.`);
    return;
  }

  const repo = `${context.repo.owner}/${context.repo.repo}`;
  await comment({ github, context, pr, body: closeMessage(repo) });
  await setState({ github, context, pr, state: "closed" });
  core.info(`Closed #${pr.number}: no accepted issue.`);
}

async function reopenForAcceptedIssue({ github, context, core }) {
  const issue = context.payload.issue;
  if (issue.pull_request || !ACCEPTED_LABELS.includes(context.payload.label.name)) return;

  const events = await github.paginate(github.rest.issues.listEventsForTimeline, {
    ...context.repo,
    issue_number: issue.number,
    per_page: 100,
  });
  const repo = `${context.repo.owner}/${context.repo.repo}`;
  const prNumbers = new Set(
    events
      .filter(
        (event) =>
          event.event === "cross-referenced" &&
          event.source?.issue?.pull_request &&
          event.source.issue.repository?.full_name === repo,
      )
      .map((event) => event.source.issue.number),
  );

  for (const number of prNumbers) {
    const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
    if (pr.state !== "closed" || pr.merged) continue;
    if (!(await wasClosedByGate({ github, context, pr }))) continue;
    const issues = await closingIssues({ github, context, pr });
    if (!issues.some((closing) => closing.number === issue.number)) continue;

    await setState({ github, context, pr, state: "open" });
    await comment({ github, context, pr, body: REOPEN_MESSAGE });
    core.info(`Reopened #${pr.number}: issue #${issue.number} was accepted.`);
  }
}

module.exports = async ({ github, context, core }) => {
  if (context.eventName === "pull_request_target") {
    await checkPullRequest({ github, context, core });
  } else if (context.eventName === "issues") {
    await reopenForAcceptedIssue({ github, context, core });
  }
};
