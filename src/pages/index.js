import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Book 📚
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Book: From Simulation to Embodied Intelligence">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <h3>ROS 2 Fundamentals</h3>
                <p>Learn the foundational concepts of Robot Operating System 2.</p>
              </div>
              <div className="col col--4">
                <h3>Simulation & AI</h3>
                <p>Master physics simulation and AI navigation techniques.</p>
              </div>
              <div className="col col--4">
                <h3>Vision-Language-Action</h3>
                <p>Integrate perception, language understanding, and action execution.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}